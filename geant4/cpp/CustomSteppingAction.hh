#ifndef CUSTOMSTEPPINGACTION_HH
#define CUSTOMSTEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "globals.hh"

class G4Step;
class G4EventManager;
class G4Event;

class CustomSteppingAction : public G4UserSteppingAction
{
public:
    CustomSteppingAction();
    virtual ~CustomSteppingAction();

    virtual void UserSteppingAction(const G4Step* step);
    void clean();

private:
    G4EventManager* eventManager;
    G4Event* event;
    int primaryTrackId;

    double killMomenta;
    bool killSecondary;

    bool store_all;
    bool store_primary;
    bool is_single_step;

public:
    // Add any necessary members here
    std::vector<double> px;

    void setStorePrimary(bool storePrimary);

    std::vector<double> py;
    std::vector<double> pz;

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;

    std::vector<double> stepLength;
    std::vector<double> chargeDeposit;
    std::vector<int> trackId;


    void setStoreAll(bool storeAll);



    void setKillMomenta(double killMomenta);

    void setKillSecondary(bool killSecondary);

    void setIsSingleStep(bool isSingleStep);

    void setMaxSteps(int maxSteps);

    double max_momenta_diff; // Only for debugging...
    int max_steps;

public:
    int num_steps;
};

#endif
