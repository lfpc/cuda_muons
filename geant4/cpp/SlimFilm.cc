//
// Created by Shah Rukh Qasim on 17.07.2024.
//

#include <G4SDManager.hh>
#include "SlimFilm.hh"
#include "DetectorConstruction.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4Sphere.hh"
#include "G4UserLimits.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VisAttributes.hh"
#include "G4FieldManager.hh"
#include "G4ChordFinder.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4Para.hh"
#include "SlimFilmSensitiveDetector.hh"


SlimFilm::SlimFilm(Json::Value detector_data) {
        detectorData = detector_data;
}

SlimFilm::~SlimFilm() noexcept {

}


G4VPhysicalVolume *SlimFilm::Construct() {
    // Get NIST material manager
    G4NistManager* nist = G4NistManager::Instance();

    // Define the world material
    G4Material* worldMaterial = nist->FindOrBuildMaterial("G4_AIR");
    // Get the world size from the JSON variable
    G4double worldSizeX = detectorData["worldSizeX"].asDouble() * m;
    G4double worldSizeY = detectorData["worldSizeY"].asDouble() * m;
    G4double worldSizeZ = detectorData["worldSizeZ"].asDouble() * m;

    G4double worldPositionX = detectorData["worldPositionX"].asDouble() * m;
    G4double worldPositionY = detectorData["worldPositionY"].asDouble() * m;
    G4double worldPositionZ = detectorData["worldPositionZ"].asDouble() * m;

    // Create the world volume
    G4Box* solidWorld = new G4Box("WorldX", worldSizeX / 2, worldSizeY / 2, worldSizeZ / 2);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, worldMaterial, "WorldY");

    G4VPhysicalVolume *physWorld = new G4PVPlacement(0, G4ThreeVector(worldPositionX, worldPositionY,
                                                                      worldPositionZ),
                                                     logicWorld, "WorldZ", 0,
                                                     false, 0, true);

    const Json::Value film = detectorData["film"];
    G4double dz = film["size_z"].asDouble() * m;
    G4double dx = film["size_x"].asDouble() * m;
    G4double dy = film["size_y"].asDouble() * m;
    G4double z_center = film["z_center"].asDouble() * m;

    G4Box* filmBox = new G4Box("Film", dx / 2, dy / 2, dz / 2);
    G4Material* boxMaterial = nist->FindOrBuildMaterial("G4_AIR");
    logicG = new G4LogicalVolume(filmBox, boxMaterial, "gggvl");

    new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), logicG, "BoxZ", logicWorld, false, 0, true);

    return physWorld;
}



void SlimFilm::setMagneticFieldValue(double strength, double theta, double phi) {
    std::cout<<"setMagneticFieldValue() not used.\n";
}

void SlimFilm::ConstructSDandField() {
    // Create the sensitive detector
    G4String sdName = "MySensitiveDetector";
    auto* sdManager = G4SDManager::GetSDMpointer();
    SlimFilmSensitiveDetector* mySD = new SlimFilmSensitiveDetector(sdName);
    sdManager->AddNewDetector(mySD);

    // Attach the sensitive detector to the logical volume
    if (logicG) {
        logicG->SetSensitiveDetector(mySD);
        std::cout<<"Sensitive set...\n";
    }
    else {
        std::cout<<"Sensitive not set, check...\n";
    }
}
