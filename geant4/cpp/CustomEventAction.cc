//
// Created by Shah Rukh Qasim on 05.07.2024.
//

#include "CustomEventAction.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4RunManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

CustomEventAction::CustomEventAction()
        : G4UserEventAction()
{
    // Constructor implementation
}

CustomEventAction::~CustomEventAction()
{
    // Destructor implementation
}

void CustomEventAction::BeginOfEventAction(const G4Event* event)
{
    if (steppingAction != nullptr) {
        steppingAction->clean();
    }
    G4int eventID = event->GetEventID();
//    G4cout << "Starting Event: " << eventID << G4endl;
    // Add additional initialization code here
}

void CustomEventAction::EndOfEventAction(const G4Event* event)
{
    G4int eventID = event->GetEventID();
//    G4cout << "Ending Event: " << eventID << G4endl;
    // Add additional finalization code here
}

CustomSteppingAction *CustomEventAction::getSteppingAction() const {
    return steppingAction;
}

void CustomEventAction::setSteppingAction(CustomSteppingAction *steppingAction) {
    CustomEventAction::steppingAction = steppingAction;
}
