//
// Created by Shah Rukh Qasim on 10.07.2024.
//

#include "GDetectorConstruction.hh"
#include "DetectorConstruction.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4Sphere.hh"
#include "G4UserLimits.hh"
#include "G4UniformMagField.hh"
#include "G4ThreeVector.hh"
#include "G4ThreeVector.hh"
#include "G4TransportationManager.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4Sphere.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4UniformMagField.hh"
#include "G4UserLimits.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VisAttributes.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4ChordFinder.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4PropagatorInField.hh"
#include "G4ClassicalRK4.hh"
#include "G4Para.hh"
#include "G4GenericTrap.hh"
#include "SlimFilmSensitiveDetector.hh"
#include "CustomMagneticField.hh"
//#include "CavernConstruction.hh"

#include <iostream>
#include <G4Trap.hh>
#include <G4GeometryTolerance.hh>
#include "H5Cpp.h"

G4VPhysicalVolume *GDetectorConstruction::Construct() {
    //#include <chrono>
    //auto start = std::chrono::high_resolution_clock::now(); //taking 5 seconds
    double limit_world_time_max_ = 5000 * ns;
    double limit_world_energy_max_ = 100 * eV;

    // Create a user limits object with a maximum step size of 1 mm
//    G4double maxStep = 5 * cm;
//    G4UserLimits* userLimits = new G4UserLimits(maxStep);

    // Define the user limits
    G4double maxTrackLength = DBL_MAX; // No limit on track length
    G4double maxStepLength = DBL_MAX;  // No limit on step length
    maxStepLength = detectorData["max_step_length"].asDouble() * m;
    maxTrackLength = DBL_MAX;
    G4double maxTime = DBL_MAX;        // No limit on time
    G4double minKineticEnergy = 100 * MeV; // Minimum kinetic energy

    // Create an instance of G4UserLimits
    G4UserLimits* userLimits2 = getLimitsFromDetectorConfig(detectorData);
    std::cout<<"Initializing Muon shield design...\n";


    // Get NIST material manager
    G4NistManager* nist = G4NistManager::Instance();
    

    // Define the world material
    G4Material* worldMaterial = nist->FindOrBuildMaterial("G4_AIR");
    // Get the world size from the JSON variable
    G4double worldSizeX = detectorData["worldSizeX"].asDouble() * m;
    G4double worldSizeY = detectorData["worldSizeY"].asDouble() * m;
    G4double worldSizeZ = detectorData["worldSizeZ"].asDouble() * m;


    // Create the world volume
    G4Box* solidWorld = new G4Box("WorldX", worldSizeX / 2, worldSizeY / 2, worldSizeZ / 2);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, worldMaterial, "WorldY");
    logicWorld->SetUserLimits(userLimits2);
    
    G4VPhysicalVolume* physWorld = new G4PVPlacement(0, G4ThreeVector(0, 0, 0), logicWorld, "WorldZ", 0, false, 0, true);
    if (detectorData.isMember("cavern")) {
        const Json::Value caverns = detectorData["cavern"];
        for (const auto& cavern : caverns){
            G4double z_center = cavern["z_center"].asDouble() * m;
            G4double dz = cavern["dz"].asDouble() * m;
            Json::Value cavern_blocks = cavern["components"];
            G4Material* boxMaterial = nist->FindOrBuildMaterial(cavern["material"].asString());
            for (auto block: cavern_blocks){
                std::vector<G4TwoVector> corners;
                for (int i = 0; i < 8; ++i) {
                corners.push_back(G4TwoVector (block[i*2].asDouble() * m, block[i*2+1].asDouble() * m));
                    }
                auto genericV = new G4GenericTrap(G4String("cavern_block"), dz, corners);
                auto logicG = new G4LogicalVolume(genericV, boxMaterial, "cavern_log");
                new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), logicG, "cavern", logicWorld, false, 0, true);
                logicG->SetUserLimits(userLimits2);
            }
        
        }
    }
    if (detectorData.isMember("target")) {
        const Json::Value targets = detectorData["target"]["components"];
        int i = 0;
        for (const auto& target : targets) {
            G4double innerRadius = 0;//target["innerRadius"].asDouble() * m;
            G4double outerRadius = target["radius"].asDouble() * m;
            G4double dz = target["dz"].asDouble() * m / 2;
            G4double startAngle = 0*deg;//target["startAngle"].asDouble() * deg;
            G4double spanningAngle = 360*deg;//target["spanningAngle"].asDouble() * deg;
            G4double z_center = target["z_center"].asDouble() * m;
            std::string materialName = target["material"].asString();
            G4Material* cylinderMaterial = nist->FindOrBuildMaterial(materialName);
            G4Tubs* solidCylinder = new G4Tubs("TargetCylinder" + std::to_string(i), innerRadius, outerRadius, dz, startAngle, spanningAngle);
            G4LogicalVolume* logicCylinder = new G4LogicalVolume(solidCylinder, cylinderMaterial, "Cylinder" + std::to_string(i));
            new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), logicCylinder, "Cylinder" + std::to_string(i), logicWorld, false, 0, true);
            logicCylinder->SetUserLimits(userLimits2);
            i++;
        }
    }
    // Process the magnets from the JSON variable
    G4MagneticField* GlobalmagField = nullptr;
    const Json::Value globalFieldMap = detectorData["global_field_map"];
    const Json::Value magnets = detectorData["magnets"];

    if (globalFieldMap.isMember("B")) {
        std::string filename = globalFieldMap["B"].asString();

        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("B");
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);

        std::vector<double> B_vector(dims[0] * dims[1]);
        dataset.read(B_vector.data(), H5::PredType::NATIVE_DOUBLE);

        if (!B_vector.empty()) {
            std::map<std::string, std::vector<double>> ranges;
            std::vector<G4ThreeVector> fields;
            ranges["range_x"] = {globalFieldMap["range_x"][0].asDouble() * cm, globalFieldMap["range_x"][1].asDouble() * cm, globalFieldMap["range_x"][2].asDouble() * cm};
            ranges["range_y"] = {globalFieldMap["range_y"][0].asDouble() * cm, globalFieldMap["range_y"][1].asDouble() * cm, globalFieldMap["range_y"][2].asDouble() * cm};
            ranges["range_z"] = {globalFieldMap["range_z"][0].asDouble() * cm, globalFieldMap["range_z"][1].asDouble() * cm, globalFieldMap["range_z"][2].asDouble() * cm};

            for (size_t i = 0; i < B_vector.size(); i += 3) {
                fields.emplace_back(B_vector[i] * tesla, B_vector[i + 1] * tesla, B_vector[i + 2] * tesla);
            }
            std::vector<double>().swap(B_vector);
            // Determine the interpolation type
            CustomMagneticField::InterpolationType interpType = CustomMagneticField::NEAREST_NEIGHBOR;
            // Define the custom magnetic field
            GlobalmagField = new CustomMagneticField(ranges, fields, interpType);
        }
    }
    //const Json::Value fields = detectorData["field_map"];
    double totalWeight = 0;
    for (const auto& magnet : magnets) {

        std::cout<<"Adding box"<<std::endl;
        // Get the material for the magnet

        G4double z_center = magnet["z_center"].asDouble() * m;
        G4double dz = magnet["dz"].asDouble() * m;

        Json::Value arb8s = magnet["components"];
        for (auto arb8: arb8s) {
            std::string materialName = arb8["material"].asString();
            G4Material* boxMaterial = nist->FindOrBuildMaterial(materialName);
            std::vector<G4TwoVector> corners_two;
            Json::Value corners = arb8["corners"];
            
            for (int i = 0; i < 8; ++i) {
                corners_two.push_back(G4TwoVector (corners[i*2].asDouble() * m, corners[i*2+1].asDouble() * m));
            }
            Json::Value field_value = arb8["field"];
            G4double fieldX;
            G4double fieldY;
            G4double fieldZ;
            G4ThreeVector fieldValue;
            G4MagneticField* magField = nullptr;
            if (arb8["field_profile"].asString() == "global") {
                magField = GlobalmagField;
            } else if (arb8["field_profile"].asString() == "uniform"){
                fieldX = field_value[0].asDouble();
                fieldY = field_value[1].asDouble();
                fieldZ = field_value[2].asDouble();
                fieldValue = G4ThreeVector(fieldX * tesla, fieldY * tesla, fieldZ * tesla);
                
                // Create and set the uniform magnetic field for the box
                magField = new G4UniformMagField(fieldValue);
            } else {
                std::map<std::string, std::vector<double>> ranges;
                std::vector<G4ThreeVector> fields;
                //const Json::Value& pointsData = field_value[0];
                //const Json::Value& fieldsData = field_value[1];
                ranges["range_x"] = {field_value["range_x"][0].asDouble() * cm, field_value["range_x"][1].asDouble() * cm, field_value["range_x"][2].asDouble() * cm};
                ranges["range_y"] = {field_value["range_y"][0].asDouble() * cm, field_value["range_y"][1].asDouble() * cm, field_value["range_y"][2].asDouble() * cm};
                ranges["range_z"] = {field_value["range_z"][0].asDouble() * cm, field_value["range_z"][1].asDouble() * cm, field_value["range_z"][2].asDouble() * cm};

                const Json::Value& fieldsData = field_value["B"];
                for (Json::ArrayIndex i = 0; i < fieldsData.size(); ++i) {
                    fields.emplace_back(fieldsData[i][0].asDouble() * tesla, fieldsData[i][1].asDouble() * tesla, fieldsData[i][2].asDouble() * tesla);
                }
                // Determine the interpolation type
                CustomMagneticField::InterpolationType interpType = CustomMagneticField::NEAREST_NEIGHBOR;
                // Define the custom magnetic field
                magField = new CustomMagneticField(ranges, fields, interpType);
            }
            auto FieldManager = new G4FieldManager();
            FieldManager->SetDetectorField(magField);
            FieldManager->CreateChordFinder(magField);
            auto genericV = new G4GenericTrap(G4String("sdf"), dz, corners_two);
            auto logicG = new G4LogicalVolume(genericV, boxMaterial, "gggvl");
            double volArb = boxMaterial->GetDensity() /(kg/m3)  * genericV->GetCubicVolume()/(m3);
            totalWeight += volArb;
            logicG->SetFieldManager(FieldManager, true);
            new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), logicG, "BoxZ", logicWorld, false, 0, true);
            logicG->SetUserLimits(userLimits2);

        }
    }
    if (GlobalmagField) {
        auto fieldManager = new G4FieldManager();
        fieldManager->SetDetectorField(GlobalmagField);
        fieldManager->CreateChordFinder(GlobalmagField);
        logicWorld->SetFieldManager(fieldManager, true);
    }


    sensitiveLogical = nullptr;
   if (detectorData.isMember("sensitive_film")) {
        int idx_plane = 0;
        const Json::Value& sensitiveFilm = detectorData["sensitive_film"];
        if (!sensitiveFilm.isArray()) {
            G4Material* air = nist->FindOrBuildMaterial("G4_AIR");

            double dz = sensitiveFilm["dz"].asDouble() * m;
            double dx = sensitiveFilm["dx"].asDouble() * m;
            double dy = sensitiveFilm["dy"].asDouble() * m;
            double z_center = sensitiveFilm["z_center"].asDouble() * m;

            auto sensitiveBox = new G4Box("sensitive_film", dx/2, dy/2, dz/2);
            sensitiveLogical = new G4LogicalVolume(sensitiveBox, air, "sensitive_film_logic");
            new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), sensitiveLogical, "sensitive_plc", logicWorld, false, 0, true);
            sensitiveLogical->SetUserLimits(userLimits2);

            auto filmSD = new SlimFilmSensitiveDetector("sensitive_film_SD", idx_plane, true); // true, since it's the only/last plane
            G4SDManager::GetSDMpointer()->AddNewDetector(filmSD);
            sensitiveLogical->SetSensitiveDetector(filmSD);
            slimFilmSensitiveDetectors.push_back(filmSD);

            std::cout<<"Sensitive film placed at "<< z_center / cm << ".\n";
        }
        else{
            G4Material* air = nist->FindOrBuildMaterial("G4_AIR");
            G4SDManager* sdManager = G4SDManager::GetSDMpointer();
            int n_planes = sensitiveFilm.size();

            // Loop through the array of film definitions in the JSON
            
            for (const auto& sensitiveFilm_i : sensitiveFilm) {
                // --- Get parameters from JSON ---
                G4String name = sensitiveFilm_i["name"].asString();
                double dz = sensitiveFilm_i["dz"].asDouble() * m;
                double dx = sensitiveFilm_i["dx"].asDouble() * m;
                double dy = sensitiveFilm_i["dy"].asDouble() * m;
                double z_center = sensitiveFilm_i["z_center"].asDouble() * m;

                // --- Create geometry (Box, Logical Volume, Placement) ---
                auto sensitiveBox = new G4Box(name + "_box", dx / 2, dy / 2, dz / 2);
                auto senslog = new G4LogicalVolume(sensitiveBox, air, name + "_logic");
                sensitiveLogicals.push_back(senslog);
                new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), senslog, name + "_plc", logicWorld, false, 0, true);
                senslog->SetUserLimits(userLimits2);

                // --- Create and register a new sensitive detector for this volume ---
                bool last = (idx_plane == (n_planes - 1));
                auto filmSD = new SlimFilmSensitiveDetector(name + "_SD", idx_plane, last);
                G4SDManager::GetSDMpointer()->AddNewDetector(filmSD);
                senslog->SetSensitiveDetector(filmSD);
                slimFilmSensitiveDetectors.push_back(filmSD);

                std::cout << "Sensitive film '" << name << "' placed at z = " << z_center / cm << " cm.\n";
                ++idx_plane;}
        }
    }

    else {
        std::cout<<"Sensitive films section not found in JSON, skipped.\n";
    }

    // --- NEW: Logic to handle a sensitive Arb8 box ---
    if (detectorData.isMember("sensitive_box")) {
    int idx_plane = -1;
    std::cout << "Constructing sensitive Arb8 box with parallel, face-based skins." << std::endl;
    const Json::Value& boxData = detectorData["sensitive_box"];
    const G4double skin_thickness = 1 * cm;
    G4String boxName = boxData["name"].asString();
    G4Material* skin_material = nist->FindOrBuildMaterial("G4_AIR");
    G4SDManager* sdManager = G4SDManager::GetSDMpointer();

    const Json::Value& xy_vertices_json = boxData["corners"];
    if (xy_vertices_json.size() != 16) {
        throw std::runtime_error("Arb8 'corners' must contain exactly 16 numbers (8 x,y pairs).");
    }
    // Store the 8 box corners, with z centered at z_center
    std::vector<G4ThreeVector> box_corners;
    double length = boxData["dz"].asDouble() * m;
    double z_center = boxData["z_center"].asDouble() * m;
    for (int i = 0; i < 8; ++i) {
        double x = xy_vertices_json[i*2].asDouble() * m;
        double y = xy_vertices_json[i*2+1].asDouble() * m;
        double z = (i < 4) ? (z_center - length) : (z_center + length);
        box_corners.emplace_back(x, y, z);
    }
    // Face definitions (indices into box_corners)
    const int faces[6][4] = {
        {0, 1, 4, 5}, // bottom
        {3, 2, 7, 6}, // top
        {0, 1, 2, 3}, // front
        {4, 5, 6, 7}, // back
        {1, 2, 5, 6}, // right
        {0, 3, 4, 7}  // left
    };
    for (int i = 0; i < 6; ++i) {
    G4String skinName = boxName + "_skin_" + std::to_string(i);
    std::vector<G4ThreeVector> face_verts = {
        box_corners[faces[i][0]],
        box_corners[faces[i][1]],
        box_corners[faces[i][2]],
        box_corners[faces[i][3]]
    };

    double half_thick = skin_thickness / 2.0;
    std::vector<G4TwoVector> trap_corners;
    G4ThreeVector center;
    double dz_trap;

    if (i == 2) { 
        continue; // Skip the front face, as per the original logic
        idx_plane = -1;
        dz_trap = half_thick;
        center = G4ThreeVector(0, 0, z_center - length);
        for (auto& v : face_verts)
            trap_corners.emplace_back(v.x(), v.y());
        for (auto& v : face_verts)
            trap_corners.emplace_back(v.x(), v.y());
    } else if (i == 3) {
        continue; // Skip the back face, as per the original logic
        idx_plane = -6;
        dz_trap = half_thick;
        center = G4ThreeVector(0, 0, z_center + length);
        for (auto& v : face_verts)
            trap_corners.emplace_back(v.x(), v.y());
        for (auto& v : face_verts)
            trap_corners.emplace_back(v.x(), v.y());
    } else if (i == 0 || i == 1) { 
        idx_plane = -2 - i; // 0 for bottom, 1 for top
        center = G4ThreeVector(0, 0, z_center);
        dz_trap = length;
        trap_corners.emplace_back(face_verts[0].x(), face_verts[0].y() - half_thick);
        trap_corners.emplace_back(face_verts[1].x(), face_verts[1].y() - half_thick);
        trap_corners.emplace_back(face_verts[1].x(), face_verts[1].y() + half_thick);
        trap_corners.emplace_back(face_verts[0].x(), face_verts[0].y() + half_thick);
        trap_corners.emplace_back(face_verts[2].x(), face_verts[2].y() - half_thick);
        trap_corners.emplace_back(face_verts[3].x(), face_verts[3].y() - half_thick);
        trap_corners.emplace_back(face_verts[3].x(), face_verts[3].y() + half_thick);
        trap_corners.emplace_back(face_verts[2].x(), face_verts[2].y() + half_thick);
    } else if (i == 4 || i == 5) { 
        idx_plane = -i; // 4 for right, 5 for left
        center = G4ThreeVector(0, 0, z_center);
        dz_trap = length;
        trap_corners.emplace_back(face_verts[0].x() - half_thick, face_verts[0].y());
        trap_corners.emplace_back(face_verts[0].x() + half_thick, face_verts[0].y());
        trap_corners.emplace_back(face_verts[1].x() + half_thick, face_verts[1].y());
        trap_corners.emplace_back(face_verts[1].x() - half_thick, face_verts[1].y());
        trap_corners.emplace_back(face_verts[2].x() - half_thick, face_verts[2].y());
        trap_corners.emplace_back(face_verts[2].x() + half_thick, face_verts[2].y());
        trap_corners.emplace_back(face_verts[3].x() + half_thick, face_verts[3].y());
        trap_corners.emplace_back(face_verts[3].x() - half_thick, face_verts[3].y());
    }

    auto skin_solid = new G4GenericTrap(skinName, dz_trap, trap_corners);
    auto skin_logical = new G4LogicalVolume(skin_solid, skin_material, skinName + "_logic");

    new G4PVPlacement(
        0, center,
        skin_logical, skinName + "_plc", logicWorld, false, 0, true
    );

    sensitiveLogicals.push_back(skin_logical);
    auto filmSD = new SlimFilmSensitiveDetector(skinName + "_SD", idx_plane, false);
    sdManager->AddNewDetector(filmSD);
    skin_logical->SetSensitiveDetector(filmSD);
    slimFilmSensitiveDetectors.push_back(filmSD);
}
    std::cout << "Placed 6 parallel, face-based sensitive skins for '" << boxName << "'." << std::endl;
    } else {
        std::cout << "Sensitive box section not found in JSON, skipped.\n";
    }
    detectorWeightTotal = totalWeight;
   return physWorld;
}



GDetectorConstruction::GDetectorConstruction(Json::Value detector_data)
    : detectorData(detector_data) {
    detectorWeightTotal = 0;
}

void GDetectorConstruction::setMagneticFieldValue(double strength, double theta, double phi) {
//    DetectorConstruction::setMagneticFieldValue(strength, theta, phi);
std::cout<<"cannot set magnetic field value for boxy detector.\n"<<std::endl;
}

double GDetectorConstruction::getDetectorWeight() {
    return detectorWeightTotal;
}


void GDetectorConstruction::ConstructSDandField()
{
}

