cmake_minimum_required(VERSION 3.12)
project(my_project)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(PYBIND11_FINDPYTHON ON)

# Find pybind11
find_package(pybind11 REQUIRED)
find_package(Geant4 REQUIRED ui_all vis_all)
find_package(HDF5 REQUIRED COMPONENTS CXX)
set(GEANT4_INCLUDE_DIR "/some/random/path" CACHE PATH "Path to Geant4 include directory")

include_directories(/usr/local/include/Geant4/)
include_directories(${GEANT4_INCLUDE_DIR})
include_directories(${HDF5_INCLUDE_DIRS})



# Add subdirectory for jsoncpp
add_subdirectory(external/jsoncpp)

# Add the include directory for jsoncpp
include_directories(external/jsoncpp/include)

set(SOURCE_FILES
        DetectorConstruction.cc
        PrimaryGeneratorAction.cc
        CustomSteppingAction.cc
        CustomEventAction.cc
        BoxyDetectorConstruction.cc
        GDetectorConstruction.cc
        SlimFilm.cc
        SlimFilmSensitiveDetector.cc
        CustomMagneticField.cc
        )


add_library(common_sources STATIC ${SOURCE_FILES})

# Add pybind11 include directories to common_sources
target_include_directories(common_sources PUBLIC ${pybind11_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})

add_executable(MuonSlab main.cc)

# Create the Python module
target_link_libraries(MuonSlab common_sources ${Geant4_LIBRARIES} jsoncpp_lib ${HDF5_CXX_LIBRARIES})
#target_link_libraries(muon_slabs ${Geant4_LIBRARIES})
set_target_properties(common_sources PROPERTIES POSITION_INDEPENDENT_CODE ON)


pybind11_add_module(muon_slabs MuonSlabs.cc)
target_link_libraries(muon_slabs PUBLIC common_sources ${Geant4_LIBRARIES} jsoncpp_lib ${HDF5_CXX_LIBRARIES})

configure_file(init_vis.mac init_vis.mac COPYONLY)
