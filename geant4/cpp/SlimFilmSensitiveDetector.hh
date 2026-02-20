//
// Created by Shah Rukh Qasim on 17.07.2024.
//

#ifndef MY_PROJECT_SLIMFILMSENSITIVEDETECTOR_HH
#define MY_PROJECT_SLIMFILMSENSITIVEDETECTOR_HH

#include <vector>

#include "G4VSensitiveDetector.hh"
#include "G4Step.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4SDManager.hh"
#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"



class SlimFilmSensitiveDetector : public G4VSensitiveDetector {
public:
    SlimFilmSensitiveDetector(const G4String& name);
    virtual ~SlimFilmSensitiveDetector();

    virtual void Initialize(G4HCofThisEvent* hce) override;
    virtual G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist) override;
    virtual void EndOfEvent(G4HCofThisEvent* hce) override;


public:
    std::vector<double> px;
    std::vector<double> py;
    std::vector<double> pz;

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;

    std::vector<int> trackId;
    std::vector<int> pid;
    std::vector<double> time;
};


#endif //MY_PROJECT_SLIMFILMSENSITIVEDETECTOR_HH
