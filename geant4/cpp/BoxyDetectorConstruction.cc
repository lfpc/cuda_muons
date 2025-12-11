//
// Created by Shah Rukh Qasim on 10.07.2024.
//

#include "BoxyDetectorConstruction.hh"
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


#include <iostream>

G4VPhysicalVolume *BoxyDetectorConstruction::Construct() {
    double limit_world_time_max_ = 5000 * ns;
    double limit_world_energy_max_ = 100 * eV;

    // Create a user limits object with a maximum step size of 1 mm
    G4double maxStep = 5 * cm;
    G4UserLimits* userLimits = new G4UserLimits(maxStep);

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
    logicWorld->SetUserLimits(userLimits);

    G4VPhysicalVolume* physWorld = new G4PVPlacement(0, G4ThreeVector(worldPositionX, worldPositionY, worldPositionZ), logicWorld, "WorldZ", 0, false, 0, true);

    // Process the components from the JSON variable
    const Json::Value components = detectorData["components"];
    for (const auto& component : components) {

        std::cout<<"Adding box"<<std::endl;
        // Get the material for the component
        std::string materialName = component["material"].asString();
        G4Material* boxMaterial = nist->FindOrBuildMaterial(materialName);

        // Get the dimensions of the box
        G4double boxSizeX = component["sizeX"].asDouble() * m;
        G4double boxSizeY = component["sizeY"].asDouble() * m;
        G4double boxSizeZ = component["sizeZ"].asDouble() * m;

        // Get the position of the box
        G4double posX = component["posX"].asDouble() * m;
        G4double posY = component["posY"].asDouble() * m;
        G4double posZ = component["posZ"].asDouble() * m;

        // Create the box volume
        G4Box* solidBox = new G4Box("BoxX", boxSizeX / 2, boxSizeY / 2, boxSizeZ / 2);
        G4LogicalVolume* logicBox = new G4LogicalVolume(solidBox, boxMaterial, "BoxY");

        // Associate the user limits with the logical volume
        logicBox->SetUserLimits(userLimits);

        new G4PVPlacement(0, G4ThreeVector(posX, posY, posZ), logicBox, "BoxZ", logicWorld, false, 0, true);

        // Get the magnetic field vector for the box
        G4double fieldX = component["fieldX"].asDouble();
        G4double fieldY = component["fieldY"].asDouble();
        G4double fieldZ = component["fieldZ"].asDouble();
        G4ThreeVector fieldValue = G4ThreeVector(fieldX * tesla, fieldY * tesla, fieldZ * tesla);

        // Create and set the magnetic field for the box
        G4UniformMagField* boxMagField = new G4UniformMagField(fieldValue);
        G4FieldManager* boxFieldManager = new G4FieldManager();
        boxFieldManager->SetDetectorField(boxMagField);

        // Create the equation of motion and the stepper for the box
//        G4Mag_UsualEqRhs* equationOfMotion = new G4Mag_UsualEqRhs(boxMagField);
//        G4MagIntegratorStepper* stepper = new G4ClassicalRK4(equationOfMotion);

        // Create the chord finder for the box
        boxFieldManager->CreateChordFinder(boxMagField);

        logicBox->SetFieldManager(boxFieldManager, true);
    }

    // Return the physical world
    return physWorld;
}



BoxyDetectorConstruction::BoxyDetectorConstruction(Json::Value detector_data) {
    detectorData = detector_data;
}

void BoxyDetectorConstruction::setMagneticFieldValue(double strength, double theta, double phi) {
//    DetectorConstruction::setMagneticFieldValue(strength, theta, phi);
std::cout<<"cannot set magnetic field value for boxy detector.\n"<<std::endl;
}
