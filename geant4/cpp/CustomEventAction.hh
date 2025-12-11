//
// Created by Shah Rukh Qasim on 05.07.2024.
//

#ifndef MUON_MATTER_CUSTOMEVENTACTION_HH
#define MUON_MATTER_CUSTOMEVENTACTION_HH


#include "G4UserEventAction.hh"
#include "globals.hh"
#include "CustomSteppingAction.hh"

class G4Event;

class CustomEventAction : public G4UserEventAction
{
public:
    CustomEventAction();
    virtual ~CustomEventAction();

    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);

private:
    CustomSteppingAction* steppingAction;
public:
    CustomSteppingAction *getSteppingAction() const;

    void setSteppingAction(CustomSteppingAction *steppingAction);
};


#endif
