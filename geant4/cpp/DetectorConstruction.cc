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
#include "SlimFilmSensitiveDetector.hh"
#include "G4SDManager.hh"


#include <iostream>

DetectorConstruction::DetectorConstruction(Json::Value detectoData)
        : G4VUserDetectorConstruction()
{
    this->detectorData = detectoData;
}

DetectorConstruction::DetectorConstruction()
: G4VUserDetectorConstruction()
{
    this->detectorData = Json::Value();
}

DetectorConstruction::~DetectorConstruction()
{ }

G4UserLimits * DetectorConstruction::getLimitsFromDetectorConfig(const Json::Value& detectorData) {
    G4double maxTrackLength = DBL_MAX; // No limit on track length
    G4double maxStepLength = DBL_MAX;  // No limit on step length
    if (not detectorData.empty()) {
        G4double temp = detectorData["limits"]["max_step_length"].asDouble() * m;
        if (temp > 0)
            maxStepLength = temp;

        std::cout<<"Applied the limit "<<temp<<std::endl<<std::endl;

    }
    maxTrackLength = DBL_MAX;
    G4double maxTime = DBL_MAX;        // No limit on time

    G4double minKineticEnergy = 100 * MeV; // Minimum kinetic energy
    if (not detectorData.empty()) {
        G4double temp = detectorData["limits"]["minimum_kinetic_energy"].asDouble() * GeV;
        std::cout<<"Applied the limit on energy "<<temp<<std::endl<<std::endl;

        if (temp > 0)
            minKineticEnergy = temp;
    }




    // Create an instance of G4UserLimits
    G4UserLimits* userLimits2 = new G4UserLimits(maxStepLength, maxTrackLength, maxTime, minKineticEnergy);
    return userLimits2;
}

G4VPhysicalVolume* DetectorConstruction::Construct() {
    G4UserLimits* userLimits2 = getLimitsFromDetectorConfig(detectorData);
    // Extract magnetic field value from detectorData["magnetic_field"]
    G4ThreeVector magnetic_field_value(0, 0, 0);
    if (!detectorData["magnetic_field"].empty() && detectorData["magnetic_field"].size() == 3) {
        magnetic_field_value = G4ThreeVector(
            detectorData["magnetic_field"][0].asDouble() * tesla,
            detectorData["magnetic_field"][1].asDouble() * tesla,
            detectorData["magnetic_field"][2].asDouble() * tesla
        );
    }

    // Get NIST material manager
    G4NistManager* nist = G4NistManager::Instance();
    std::string material_name = detectorData.get("material", "G4_Fe").asString();
    // Define the material
    G4Material* sphereMaterial = nist->FindOrBuildMaterial(material_name);
    std::cout << "Placing gigantic sphere: " << *sphereMaterial << std::endl;


    // Define the radius of the sphere
    G4double sphereRadius = 500 * m;

    // Define the world volume
    G4double worldSizeXY = 1.2 * sphereRadius * 2;
    G4double worldSizeZ  = 1.2 * sphereRadius * 2;
    G4Material* worldMaterial = nist->FindOrBuildMaterial("G4_AIR");

    // Create the world volume
    G4Box* solidWorld = new G4Box("WorldX", worldSizeXY / 2, worldSizeXY / 2, worldSizeZ / 2);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, worldMaterial, "WorldY");
    logicWorld->SetUserLimits(userLimits2);

    G4VPhysicalVolume* physWorld = new G4PVPlacement(0, G4ThreeVector(), logicWorld, "WorldZ", 0, false, 0, true);

    // Create the iron sphere
    G4Sphere* solidSphere = new G4Sphere("SphereX", 0, sphereRadius, 0, 360 * deg, 0, 180 * deg);
//    G4Box* solidSphere = new G4Box("WorldX", sphereRadius, sphereRadius, sphereRadius);

    G4LogicalVolume* logicSphere = new G4LogicalVolume(solidSphere, sphereMaterial, "SphereY");


    // Associate the user limits with a logical volume
    logicSphere->SetUserLimits(userLimits2);


    new G4PVPlacement(0, G4ThreeVector(), logicSphere, "SphereZ", logicWorld, false, 0, true);

    // Define the uniform magnetic field
    magField = new G4UniformMagField(magnetic_field_value);

    // Get the global field manager
    G4FieldManager* fieldManager = G4TransportationManager::GetTransportationManager()->GetFieldManager();

    // Set the magnetic field to the field manager
    fieldManager->SetDetectorField(magField);

    G4Mag_UsualEqRhs* equationOfMotion = new G4Mag_UsualEqRhs(magField);
    G4MagIntegratorStepper* stepper = new G4ClassicalRK4(equationOfMotion);

    fieldManager->CreateChordFinder(magField);

    logicWorld->SetFieldManager(fieldManager, true);

    sensitiveLogical = nullptr;
    if (detectorData.isMember("sensitive_film")) {
        const Json::Value sensitiveFilm = detectorData["sensitive_film"];
        std::string shape = sensitiveFilm["shape"].asString();
        if (shape == "sphere") {
            G4double radius = sensitiveFilm["radius"].asDouble() * m;
            G4double dr = sensitiveFilm["dr"].asDouble() * m;
            G4double z_center = sensitiveFilm["z_center"].asDouble() * m;

        G4double innerRadius = radius - dr;
        if (innerRadius < 0) {
            innerRadius = 0;
        }

        auto sensitiveSphere = new G4Sphere("sensitive_sphere_solid",
                                            innerRadius,    // Inner radius (pRmin)
                                            radius,         // Outer radius (pRmax)
                                            0.,             // Starting Phi angle
                                            2. * M_PI,      // Segment angle in Phi
                                            0.,             // Starting Theta angle
                                            M_PI);          // Segment angle in Theta
        sensitiveLogical = new G4LogicalVolume(sensitiveSphere, sphereMaterial, "sensitive_sphere_logic");
        new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), sensitiveLogical, "sensitive_sphere_plc", logicSphere, false, 0, true);
        std::cout << "Sensitive sphere placed successfully.\n";

    } else if (shape == "plane") {
        double dz = sensitiveFilm["dz"].asDouble() * m;
        double dx = sensitiveFilm["dx"].asDouble() * m;
        double dy = sensitiveFilm["dy"].asDouble() * m;
        double z_center = sensitiveFilm["z_center"].asDouble() * m;

        auto sensitiveBox = new G4Box("sensitive_film", dx/2, dy/2, dz/2);
        sensitiveLogical = new G4LogicalVolume(sensitiveBox, sphereMaterial, "sensitive_film_logic");
        new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), sensitiveLogical, "sensitive_plc", logicSphere, false, 0, true);
        std::cout << "Sensitive plane placed successfully.\n";

    } else {
        G4Exception("DetectorConstruction::Construct", "InvalidFormat", FatalException,
                    "Sensitive film 'format' must be 'sphere' or 'plane'.");
    }
    }
    else {
        std::cout << "Sensitive films section not found in JSON, skipped.\n";
    }
    return physWorld;
}

void DetectorConstruction::setMagneticFieldValue(double x, double y, double z) {
    G4ThreeVector fieldValue = G4ThreeVector(x*tesla, y*tesla, z*tesla);

    magField->SetFieldValue(fieldValue);
}

double DetectorConstruction::getDetectorWeight() {
    return -1;
}

void DetectorConstruction::ConstructSDandField() {
    G4VUserDetectorConstruction::ConstructSDandField();

    // Attach the sensitive detector to the logical volume
    if (sensitiveLogical) {
        auto* sdManager = G4SDManager::GetSDMpointer();

        G4String sdName = "MySensitiveDetector";
        if (!slimFilmSensitiveDetector) {
            slimFilmSensitiveDetector = new SlimFilmSensitiveDetector(sdName);
            sdManager->AddNewDetector(slimFilmSensitiveDetector);
        }
        sensitiveLogical->SetSensitiveDetector(slimFilmSensitiveDetector);
        std::cout<<"Sensitive set...\n";
    }

}
