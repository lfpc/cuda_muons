//
// Created by Shah Rukh Qasim on 10.07.2024.
//

#ifndef MY_PROJECT_GDETECTORCONSTRUCTION_HH
#define MY_PROJECT_GDETECTORCONSTRUCTION_HH

#include "DetectorConstruction.hh"
#include "json/json.h"
#include "SlimFilmSensitiveDetector.hh"

class GDetectorConstruction : public DetectorConstruction {
public:
    virtual G4VPhysicalVolume *Construct();
public:
    GDetectorConstruction(Json::Value detector_data);
protected:
    Json::Value detectorData;
public:
    void ConstructSDandField() override;

protected:
    double detectorWeightTotal;
    G4LogicalVolume* sensitiveLogical;
    std::vector<G4LogicalVolume*> sensitiveLogicals;
public:
    double getDetectorWeight() override;
    void setMagneticFieldValue(double strength, double theta, double phi) override;

};


#endif //MY_PROJECT_GDETECTORCONSTRUCTION_HH
