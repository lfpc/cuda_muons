#include "CustomSteppingAction.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4RunManager.hh"
#include "G4ios.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4ChordFinder.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4PropagatorInField.hh"
#include "G4ClassicalRK4.hh"


#include <iostream>

CustomSteppingAction::CustomSteppingAction()
    : G4UserSteppingAction(), eventManager(G4EventManager::GetEventManager())
{
    primaryTrackId = 1; // Assume it's one; might need to be changed if more than one primary particles are introduced
    killMomenta = -1;
    max_momenta_diff = -1;
    killSecondary = false;
    store_all = false;
    store_primary = false;
    is_single_step = false;
    max_steps = -1;
}

CustomSteppingAction::~CustomSteppingAction()
{}

void CustomSteppingAction::UserSteppingAction(const G4Step* step)
{

//    std::cout<<"Hello from the CustomSteppingAction::UserSteppingAction!\n";
    // Get the track
    G4Track* track = step->GetTrack();

    // Get the volume of the current step
    G4VPhysicalVolume* volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();

    // Get the kinetic energy of the particle
    G4double kineticEnergy = step->GetPreStepPoint()->GetKineticEnergy();

    // Get the energy deposited in this step
    G4double edep = step->GetTotalEnergyDeposit();

    // Get the position of the step
    G4ThreeVector pos = step->GetPostStepPoint()->GetPosition();

    num_steps += 1;

    G4StepPoint* preStepPoint = step->GetPreStepPoint();
    G4ThreeVector positiont = preStepPoint->GetPosition();
    G4double position[4];  // [Bx, By, Bz, Ex, Ey, Ez]
    position[0] = positiont[0];
    position[1] = positiont[1];
    position[2] = positiont[2];


    // Get the global field manager
    G4FieldManager* fieldManager = G4TransportationManager::GetTransportationManager()->GetFieldManager();

    // Define a vector to hold the field value
    G4ThreeVector fieldValue;

    G4ThreeVector momentum = track->GetMomentum();



    if ((store_primary and track->GetTrackID() == primaryTrackId) or store_all) {
        G4ThreeVector position2 = track->GetPosition();

        // Fill the vectors with current step data
        px.push_back(momentum.x() / GeV);
        py.push_back(momentum.y() / GeV);
        pz.push_back(momentum.z() / GeV);

        x.push_back(position2.x() / m);
        y.push_back(position2.y() / m);
        z.push_back(position2.z() / m);
        trackId.push_back(track->GetTrackID());

        stepLength.push_back(step->GetStepLength() / m);
        chargeDeposit.push_back(step->GetTotalEnergyDeposit());
    }
    if ((killSecondary && track->GetTrackID() != primaryTrackId) or is_single_step) {
        track->SetTrackStatus(fStopAndKill);
    }
    else {

    }

    if (killMomenta > 0) {
        if (momentum.mag() / GeV < killMomenta) {
//                std::cout<<"Killing because found moments is "<<momentum.mag() / GeV<<" GeV and to be killed at "<<killMomenta<<"\n";
            track->SetTrackStatus(fStopAndKill);
        }
    }

    if (max_steps != -1 and num_steps >= max_steps) {
        track->SetTrackStatus(fStopAndKill);
    }




    // Get the G4ParticleDefinition from the G4Track
    const G4ParticleDefinition* particleDef = track->GetDefinition();

    // Retrieve the PDG ID
    G4int pdgID = particleDef->GetPDGEncoding();

    // Now you can use pdgID as needed
//    std::cout << "XX: " << pdgID << " ";

//   std::cout<<track->GetKineticEnergy()/GeV<<std::endl;
}


void CustomSteppingAction::clean() {
    px.clear();
    py.clear();
    pz.clear();
    x.clear();
    y.clear();
    z.clear();
    stepLength.clear();
    chargeDeposit.clear();
    trackId.clear();
//    std::cout<<"Cleaning!"<<std::endl;
}

void CustomSteppingAction::setKillMomenta(double killMomenta) {
    CustomSteppingAction::killMomenta = killMomenta;
}

void CustomSteppingAction::setKillSecondary(bool killSecondary) {
    CustomSteppingAction::killSecondary = killSecondary;
}

void CustomSteppingAction::setStoreAll(bool storeAll) {
    store_all = storeAll;
}

void CustomSteppingAction::setStorePrimary(bool storePrimary) {
    store_primary = storePrimary;
}

void CustomSteppingAction::setIsSingleStep(bool isSingleStep) {
    is_single_step = isSingleStep;
}

void CustomSteppingAction::setMaxSteps(int maxSteps) {
    max_steps = maxSteps;
}
