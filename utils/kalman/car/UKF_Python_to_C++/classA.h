#ifndef CLASSA_H
#define CLASSA_H

#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;


class MeasurementPackage {
public:
  long long timestamp_;

  enum SensorType{
    LASER,
    RADAR
  } sensor_type_;

  Eigen::VectorXd raw_measurements_;

};

class classA {

	public:

    // Laser measurement noise standard deviation position1 in m
    const double std_laspx_ = 0.1;

    // Laser measurement noise standard deviation position2 in m
    const double std_laspy_ = 0.1;

    // Radar measurement noise standard deviation radius in m
    const double std_radr_ = 1.5;

    // Radar measurement noise standard deviation angle in rad
    const double std_radphi_ = 0.3;

    // Radar measurement noise standard deviation radius change in m/s
    const double std_radrd_ = 0.1;


    // initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    VectorXd x_;

    // state covariance matrix
    MatrixXd P_;

    // predicted sigma points matrix
    MatrixXd Xsig_pred_;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    // Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    // Weights of sigma points
    VectorXd weights_;

    // State dimension
    const int n_x_ = 5;

    // Augmented state dimension
    const int n_aug_ = 7;

    // Sigma point spreading parameter
    const double lambda_ = 3 - n_aug_;


    MeasurementPackage meas_package;

    // constructor
    classA();

    virtual ~classA();

    // methods

    void predict(const void * predicted_state, const void * timeDiff);

    void initializeStateVector(const void * meas_package_input);

    MatrixXd calculateSigmaPoints();

    void passSigmaPointsToProcessFunc(MatrixXd Xsig_aug, const void * timeDiff);

    void predictFinal(const void * predicted_state);

    void update(const void * meas_package_input, const void * updated_state);
    void updateLidar(const void * updated_state);
    void updateRadar(const void * updated_state);

    void normalizeAngle(double *angle);

};

#endif
