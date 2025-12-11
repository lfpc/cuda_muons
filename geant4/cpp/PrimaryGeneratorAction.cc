#include "PrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include <iostream>

PrimaryGeneratorAction::PrimaryGeneratorAction()
: G4VUserPrimaryGeneratorAction()
{

//    G4int n_particle = 1;
//    fParticleGun = new G4ParticleGun(n_particle);
//
//    // Default particle kinematics
//    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
//    G4String particleName = "mu+";
//    G4ParticleDefinition* particle = particleTable->FindParticle(particleName);
//
//    fParticleGun->SetParticleDefinition(particle);
//    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.,0.,1.));
//    fParticleGun->SetParticleEnergy(50.0*GeV);
////    fParticleGun->SetParticlePosition(G4ThreeVector(0.,0.,-0.5*m));
//    fParticleGun->SetParticlePosition(G4ThreeVector(0.,0.,0));
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
    if(m_steppingAction!=NULL) {
//        std::cout<<"Num steps last run: "<<m_steppingAction->num_steps<<std::endl;
        m_steppingAction->num_steps = 0;
    }
    else {
        std::cout<<"Problem!"<<std::endl;
    }
//    fParticleGun->GeneratePrimaryVertex(anEvent);
//    std::cout<<"Hello from the PrimaryGeneratorAction::GeneratePrimaries!\n";

    // Define particle properties
    G4String particleName = "mu-";

    if(next_charge == 1)
        particleName = "mu+";

    G4ThreeVector position(next_x*m, next_y*m, next_z*m);
    G4ThreeVector momentum(next_px*GeV, next_py*GeV, next_pz*GeV);
    G4double time = 0;
    // Get particle definition from G4ParticleTable
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* particleDefinition = particleTable->FindParticle(particleName);

//    std::cout << "Charge: " << particleDefinition->GetPDGCharge () << std::endl;

    if ( ! particleDefinition ) {
    G4cerr << "Error: " << particleName << " not found in G4ParticleTable" << G4endl;
    exit(1);
    }
    // Create primary particle
    G4PrimaryParticle* primaryParticle = new G4PrimaryParticle(particleDefinition);
    primaryParticle->SetMomentum(momentum.x(), momentum.y(), momentum.z());

    //std::cout<<"MMM: "<<primaryParticle->GetTotalMomentum() / GeV<<std::endl;

//    primaryParticle->SetParticleEnergy(2000. * GeV);

    // Create vertex
    G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time);
    vertex->SetPrimary(primaryParticle);
    anEvent->AddPrimaryVertex(vertex);
}

void PrimaryGeneratorAction::setSteppingAction(CustomSteppingAction* steppingAction) {
    m_steppingAction = steppingAction;
}


void PrimaryGeneratorAction::setNextMomenta(double nextPx, double nextPy, double nextPz) {
    next_px = nextPx;
    next_py = nextPy;
    next_pz = nextPz;
}


void PrimaryGeneratorAction::setNextPosition(double nextX, double nextY, double nextZ) {
    next_x = nextX;
    next_y = nextY;
    next_z = nextZ;
}

void PrimaryGeneratorAction::setNextCharge(int charge) {
    PrimaryGeneratorAction::next_charge = charge;
}
