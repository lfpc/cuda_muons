#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include <G4UniformMagField.hh>
#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "json/json.h"
#include "G4UserLimits.hh"
#include "SlimFilmSensitiveDetector.hh"

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    DetectorConstruction(Json::Value detectorData);
    virtual ~DetectorConstruction();

    virtual G4VPhysicalVolume* Construct();
    std::vector<SlimFilmSensitiveDetector*> slimFilmSensitiveDetectors;
    virtual void setMagneticFieldValue(double x, double y, double z);
    virtual G4UserLimits * getLimitsFromDetectorConfig(const Json::Value& detectorData);
    virtual double getDetectorWeight();
    void ConstructSDandField() override;
protected:
    G4UniformMagField* magField;
    G4LogicalVolume* sensitiveLogical;
    Json::Value detectorData;
};

#endif
