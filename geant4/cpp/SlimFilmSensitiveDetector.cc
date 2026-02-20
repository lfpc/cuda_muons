//
// Created by Shah Rukh Qasim on 17.07.2024.
//

#include "SlimFilmSensitiveDetector.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"


SlimFilmSensitiveDetector::SlimFilmSensitiveDetector(const G4String &name)
    : G4VSensitiveDetector(name) {}

SlimFilmSensitiveDetector::~SlimFilmSensitiveDetector() {}

void SlimFilmSensitiveDetector::Initialize(G4HCofThisEvent *hce) {
    px.clear();
    py.clear();
    pz.clear();

    x.clear();
    y.clear();
    z.clear();

    trackId.clear();
    pid.clear();
    time.clear();

}

G4bool SlimFilmSensitiveDetector::ProcessHits(G4Step *aStep, G4TouchableHistory *ROhist) {
    auto theTrack = aStep->GetTrack();
    
    auto momentum = theTrack->GetMomentum();
    auto position2 = theTrack->GetPosition();

    trackId.push_back(theTrack->GetTrackID());
    // Fill the vectors with current step data
    px.push_back(momentum.x() / GeV);
    py.push_back(momentum.y() / GeV);
    pz.push_back(momentum.z() / GeV);

    x.push_back(position2.x() / m);
    y.push_back(position2.y() / m);
    z.push_back(position2.z() / m);

    pid.push_back(theTrack->GetDefinition()->GetPDGEncoding());

    double t_ns = aStep->GetPreStepPoint()->GetGlobalTime() / ns;
    time.push_back(t_ns);

    return true;
}

void SlimFilmSensitiveDetector::EndOfEvent(G4HCofThisEvent *hce) {


}