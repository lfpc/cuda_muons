#ifndef PrimaryGeneratorAction_h
#define PrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4Event.hh"
#include "CustomSteppingAction.hh"

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
    PrimaryGeneratorAction();
    virtual ~PrimaryGeneratorAction();

    virtual void GeneratePrimaries(G4Event*);
    void setSteppingAction(CustomSteppingAction* steppingAction);

private:
    G4ParticleGun* fParticleGun;
    double next_px;
    double next_py;
    double next_pz;
    double next_x;
    double next_y;
    double next_z;
    int next_charge;
public:
    void setNextMomenta(double nextPx, double nextPy, double nextPz);
    void setNextPosition(double nextX, double nextY, double nextZ);

protected:
public:
    void setNextCharge(int charge);

protected:
    CustomSteppingAction * m_steppingAction;
};

#endif
