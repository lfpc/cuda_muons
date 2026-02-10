#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "QBBC.hh"
#include "G4UImanager.hh"
#include "FTFP_BERT.hh"
#include "QGSP_BERT_HP.hh"
#include "QGSP_BERT.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"

#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "CustomSteppingAction.hh"

#include "G4PhysicsListHelper.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4MuBremsstrahlung.hh"
#include "G4PhysListFactory.hh"
#include "G4ParticleTypes.hh"



int main(int argc, char** argv)
{
    // Detect interactive mode (if no arguments) and define UI session
    G4UIExecutive* ui = nullptr;
    if (argc == 1) {
        ui = new G4UIExecutive(argc, argv);
    }

    // Construct the default run manager
    G4RunManager* runManager = new G4RunManager;

//    std::ifstream inputFile("../../data/boxy.json");
    std::ifstream inputFile("../../data/gdetector.json");
    if (!inputFile) {
        std::cerr << "Unable to open file";
        return 1;
    }

    std::string fileContents;
    std::string line;

    while (std::getline(inputFile, line)) {
        fileContents += line + "\n";
    }

    inputFile.close();

    Json::Value detectorData;
    Json::CharReaderBuilder readerBuilder;
    std::string errs;

    std::istringstream iss(fileContents);
    if (Json::parseFromStream(readerBuilder, iss, &detectorData, &errs)) {
        // Output the parsed JSON object
        std::cout << detectorData["worldSizeX"] << std::endl;
    } else {
        std::cerr << "Failed to parse JSON: " << errs << std::endl;
    }


    // Set mandatory initialization classes
    runManager->SetUserInitialization(new GDetectorConstruction(fileContents));
//    runManager->SetUserInitialization(new BoxyDetectorConstruction(fileContents));
//    runManager->SetUserInitialization(new DetectorConstruction);


    // Use the QGSP_BERT physics list
    G4VModularPhysicsList* physicsList = new QGSP_BERT;
//    physicsList->RegisterPhysics(new G4EmLivermorePhysics());
//    physicsList->RegisterPhysics(new G4StepLimiterPhysics());
    runManager->SetUserInitialization(physicsList);

    PrimaryGeneratorAction *primaryGeneratorAction = new PrimaryGeneratorAction();
    CustomSteppingAction *customSteppingAction = new CustomSteppingAction();
    primaryGeneratorAction->setSteppingAction(customSteppingAction);

    // Set user action classes
    runManager->SetUserAction(primaryGeneratorAction);
    runManager->SetUserAction(customSteppingAction);

    // Initialize visualization
    G4VisManager* visManager = new G4VisExecutive;
    visManager->Initialize();

    // Get the pointer to the User Interface manager
    G4UImanager* UImanager = G4UImanager::GetUIpointer();

    if (ui) {
        // Interactive mode
        UImanager->ApplyCommand("/control/execute init_vis.mac");
        ui->SessionStart();
        delete ui;
    } else

    {
        // Batch mode
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        UImanager->ApplyCommand(command+fileName);
    }

    // Job termination
    delete visManager;
    delete runManager;
    return 0;
}