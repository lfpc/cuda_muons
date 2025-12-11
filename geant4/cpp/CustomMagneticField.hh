#include <vector>
#include <map>
#include "G4ThreeVector.hh"
#include "G4MagneticField.hh"

class CustomMagneticField : public G4MagneticField {
public:
    enum InterpolationType { NEAREST_NEIGHBOR, LINEAR };
    CustomMagneticField(const std::map<std::string, std::vector<double>>& ranges, const std::vector<G4ThreeVector>& fields, InterpolationType interpType);
    ~CustomMagneticField();

    void GetFieldValue(const G4double Point[4], G4double *Bfield) const override;
    void GetFieldValueNearestNeighbor(const G4double Point[4], G4double *Bfield) const;
    void GetFieldValueLinear(const G4double Point[4], G4double *Bfield) const;

private:
    std::vector<G4ThreeVector> fFields;
    InterpolationType fInterpType;

    // Grid parameters
    double x_min, x_max, dx_inv;
    double y_min, y_max, dy_inv;
    double z_min, z_max, dz_inv;
    int nx, ny, nz;

    void initializeGrid(const std::map<std::string, std::vector<double>>& ranges);
};