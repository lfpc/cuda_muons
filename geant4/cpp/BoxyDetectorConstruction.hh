//
// Created by Shah Rukh Qasim on 10.07.2024.
//

#ifndef MY_PROJECT_BOXYDETECTORCONSTRUCTION_HH
#define MY_PROJECT_BOXYDETECTORCONSTRUCTION_HH

#include "DetectorConstruction.hh"
#include "json/json.h"
#include "G4UserLimits.hh"

class BoxyDetectorConstruction : public DetectorConstruction {
public:
    virtual G4VPhysicalVolume *Construct();
public:
    BoxyDetectorConstruction(Json::Value detector_data);
protected:
    Json::Value detectorData;
public:
    void setMagneticFieldValue(double strength, double theta, double phi) override;

};


#endif //MY_PROJECT_BOXYDETECTORCONSTRUCTION_HH
