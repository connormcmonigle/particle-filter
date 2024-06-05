#pragma once

#include <Eigen/Dense>

namespace fast_plate_orbit {

struct observed_plate_orbit {
  float radius;
  float orientation;
  Eigen::Vector3f center;
};

}
