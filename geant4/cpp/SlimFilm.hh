//
// Created by Shah Rukh Qasim on 17.07.2024.
//

#ifndef MY_PROJECT_SLIMFILM_HH
#define MY_PROJECT_SLIMFILM_HH


#include <G4UniformMagField.hh>
#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "json/json.h"
#include "DetectorConstruction.hh"
#include <G4VSensitiveDetector.hh>


class SlimFilm : public DetectorConstruction
{
public:
    SlimFilm(Json::Value detector_data);
    virtual ~SlimFilm();

    virtual G4VPhysicalVolume* Construct();
    virtual void setMagneticFieldValue(double strength, double theta, double phi) override;
    void ConstructSDandField() override;

protected:
    G4UniformMagField* magField;
    Json::Value detectorData;
    G4LogicalVolume* logicG;

};


#endif //MY_PROJECT_SLIMFILM_HH
