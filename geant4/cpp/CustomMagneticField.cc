#include "CustomMagneticField.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

CustomMagneticField::CustomMagneticField(const std::map<std::string, std::vector<double>>& ranges, const std::vector<G4ThreeVector>& fields, InterpolationType interpType)
    : fFields(fields), fInterpType(interpType) {
    // Initialize grid parameters
    initializeGrid(ranges);
}

CustomMagneticField::~CustomMagneticField() {
}

void CustomMagneticField::initializeGrid(const std::map<std::string, std::vector<double>>& ranges) {
    x_min = ranges.at("range_x")[0];
    x_max = ranges.at("range_x")[1];
    dx_inv = 1.0 / ranges.at("range_x")[2];

    y_min = ranges.at("range_y")[0];
    y_max = ranges.at("range_y")[1];
    dy_inv = 1.0 / ranges.at("range_y")[2];

    z_min = ranges.at("range_z")[0];
    z_max = ranges.at("range_z")[1];
    dz_inv = 1.0 / ranges.at("range_z")[2];

    nx = static_cast<int>(std::round((x_max - x_min) * dx_inv))+1;
    ny = static_cast<int>(std::round((y_max - y_min) * dy_inv))+1;
    nz = static_cast<int>(std::round((z_max - z_min) * dz_inv))+1;
    if (fFields.size() != static_cast<size_t>(nx * ny * nz)) {
        throw std::runtime_error("Size of fFields does not match grid dimensions (nx*ny*nz)");
    }

    std::cout << "Grid initialized with dimensions: " << nx << " x " << ny << " x " << nz << std::endl;
}

void CustomMagneticField::GetFieldValueNearestNeighbor(const G4double Point[4], G4double *Bfield) const {
    // Check if the point is outside the grid
    if (fabs(Point[0]) > x_max || fabs(Point[1]) > y_max || fabs(Point[2]) > z_max) {
        Bfield[0] = 0.0;
        Bfield[1] = 0.0;
        Bfield[2] = 0.0;
        return;
    }
    // Determine the quadrant of the point
    int quadrant = 0;
    if (Point[0] >= 0 && Point[1] >= 0) {
        quadrant = 1;
    } else if (Point[0] < 0 && Point[1] >= 0) {
        quadrant = 2;
    } else if (Point[0] < 0 && Point[1] < 0) {
        quadrant = 3;
    } else {
        quadrant = 4;
    }
    // Create a copy of Point and give it the value of the corresponding symmetry to the 1st quadrant
    G4double SymmetricPoint[4] = {Point[0], Point[1], Point[2], Point[3]};
    if (quadrant == 2) {
        SymmetricPoint[0] *= -1;
    } else if (quadrant == 3) {
        SymmetricPoint[0] *= -1;
        SymmetricPoint[1] *= -1;
    } else if (quadrant == 4) {
        SymmetricPoint[1] *= -1;
    }

    // Calculate nearest indices using integer arithmetic
    int i = (int)round((SymmetricPoint[0] - x_min) * dx_inv);
    int j = (int)round((SymmetricPoint[1] - y_min) * dy_inv);
    int k = (int)round((SymmetricPoint[2] - z_min) * dz_inv);

    // Check bounds
    if (i < 0 || i >nx || j < 0 || j >ny || k < 0 || k > nz) {
        Bfield[0] = Bfield[1] = Bfield[2] = 0.0;
        return; // Out of bounds
    }

    // Compute flat index
    int idx = j*(nx*nz)+i*nz+k; //indexing of the field must match this

    // Assign the nearest values
    Bfield[0] = fFields[idx].x();
    Bfield[1] = fFields[idx].y();
    Bfield[2] = fFields[idx].z();

    // Apply symmetry to the magnetic field
    if (quadrant == 2 || quadrant == 4) {
        Bfield[0] = -Bfield[0];
    }
    if (quadrant == 3 || quadrant == 4) {
        Bfield[2] = -Bfield[2];
    }
}

void CustomMagneticField::GetFieldValueLinear(const G4double Point[4], G4double *Bfield) const {
    // TODO
}

void CustomMagneticField::GetFieldValue(const G4double Point[4], G4double *Bfield) const {
    if (fInterpType == NEAREST_NEIGHBOR) {
        GetFieldValueNearestNeighbor(Point, Bfield);
    } else {
        GetFieldValueLinear(Point, Bfield);
    }
}