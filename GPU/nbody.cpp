#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

double G = 6.674 * std::pow(10, -11);
// double G = 1;

struct simulation {
  size_t nbpart;

  std::vector<double> mass;

  // position
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  // velocity
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;

  // force
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;

  simulation(size_t nb)
      : nbpart(nb), mass(nb), x(nb), y(nb), z(nb), vx(nb), vy(nb), vz(nb),
        fx(nb), fy(nb), fz(nb) {}
};

struct device_simulation {
  double *mass;
  double *x, *y, *z;
  double *vx, *vy, *vz;
  double *fx, *fy, *fz;
};

void random_init(simulation &s) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution dismass(0.9, 1.);
  std::normal_distribution dispos(0., 1.);
  std::normal_distribution disvel(0., 1.);

  for (size_t i = 0; i < s.nbpart; ++i) {
    s.mass[i] = dismass(gen);

    s.x[i] = dispos(gen);
    s.y[i] = dispos(gen);
    s.z[i] = dispos(gen);
    s.z[i] = 0.;

    s.vx[i] = disvel(gen);
    s.vy[i] = disvel(gen);
    s.vz[i] = disvel(gen);
    s.vz[i] = 0.;
    s.vx[i] = s.y[i] * 1.5;
    s.vy[i] = -s.x[i] * 1.5;
  }

  return;
  // normalize velocity (using normalization found on some physicis blog)
  double meanmass = 0;
  double meanmassvx = 0;
  double meanmassvy = 0;
  double meanmassvz = 0;
  for (size_t i = 0; i < s.nbpart; ++i) {
    meanmass += s.mass[i];
    meanmassvx += s.mass[i] * s.vx[i];
    meanmassvy += s.mass[i] * s.vy[i];
    meanmassvz += s.mass[i] * s.vz[i];
  }
  for (size_t i = 0; i < s.nbpart; ++i) {
    s.vx[i] -= meanmassvx / meanmass;
    s.vy[i] -= meanmassvy / meanmass;
    s.vz[i] -= meanmassvz / meanmass;
  }
}

void init_solar(simulation &s) {
  enum Planets {
    SUN,
    MERCURY,
    VENUS,
    EARTH,
    MARS,
    JUPITER,
    SATURN,
    URANUS,
    NEPTUNE,
    MOON
  };
  s = simulation(10);

  // Masses in kg
  s.mass[SUN] = 1.9891 * std::pow(10, 30);
  s.mass[MERCURY] = 3.285 * std::pow(10, 23);
  s.mass[VENUS] = 4.867 * std::pow(10, 24);
  s.mass[EARTH] = 5.972 * std::pow(10, 24);
  s.mass[MARS] = 6.39 * std::pow(10, 23);
  s.mass[JUPITER] = 1.898 * std::pow(10, 27);
  s.mass[SATURN] = 5.683 * std::pow(10, 26);
  s.mass[URANUS] = 8.681 * std::pow(10, 25);
  s.mass[NEPTUNE] = 1.024 * std::pow(10, 26);
  s.mass[MOON] = 7.342 * std::pow(10, 22);

  // Positions (in meters) and velocities (in m/s)
  double AU = 1.496 * std::pow(10, 11); // Astronomical Unit

  s.x = {0,          0.39 * AU,
         0.72 * AU,  1.0 * AU,
         1.52 * AU,  5.20 * AU,
         9.58 * AU,  19.22 * AU,
         30.05 * AU, 1.0 * AU + 3.844 * std::pow(10, 8)};
  s.y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  s.vx = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.vy = {0, 47870, 35020, 29780, 24130, 13070, 9680, 6800, 5430, 29780 + 1022};
  s.vz = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

// meant to update the force that from applies on to
__global__ void compute_forces_kernel(size_t nbpart, const double *mass,
                                      const double *x, const double *y,
                                      const double *z, double *fx, double *fy,
                                      double *fz) {
  size_t to = blockIdx.x * blockDim.x + threadIdx.x;
  if (to >= nbpart)
    return;

  const double softening = .1;
  double fxi = 0.;
  double fyi = 0.;
  double fzi = 0.;

  for (size_t from = 0; from < nbpart; ++from) {
    if (from == to)
      continue;

    double dx = x[from] - x[to];
    double dy = y[from] - y[to];
    double dz = z[from] - z[to];
    double dist_sq = dx * dx + dy * dy + dz * dz;
    double norm = sqrt(dist_sq + softening);
    double F = 6.674e-11 * mass[from] * mass[to] / (dist_sq + softening);

    fxi += dx / norm * F;
    fyi += dy / norm * F;
    fzi += dz / norm * F;
  }

  fx[to] = fxi;
  fy[to] = fyi;
  fz[to] = fzi;
}

void reset_force(simulation &s) {
  for (size_t i = 0; i < s.nbpart; ++i) {
    s.fx[i] = 0.;
    s.fy[i] = 0.;
    s.fz[i] = 0.;
  }
}

__global__ void move_kernel(size_t nbpart, double dt, const double *mass,
                            double *x, double *y, double *z, double *vx,
                            double *vy, double *vz, const double *fx,
                            const double *fy, const double *fz) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nbpart)
    return;

  vx[i] += fx[i] / mass[i] * dt;
  vy[i] += fy[i] / mass[i] * dt;
  vz[i] += fz[i] / mass[i] * dt;

  x[i] += vx[i] * dt;
  y[i] += vy[i] * dt;
  z[i] += vz[i] * dt;
}

void dump_state(simulation &s) {
  std::cout << s.nbpart << '\t';
  for (size_t i = 0; i < s.nbpart; ++i) {
    std::cout << s.mass[i] << '\t';
    std::cout << s.x[i] << '\t' << s.y[i] << '\t' << s.z[i] << '\t';
    std::cout << s.vx[i] << '\t' << s.vy[i] << '\t' << s.vz[i] << '\t';
    std::cout << s.fx[i] << '\t' << s.fy[i] << '\t' << s.fz[i] << '\t';
  }
  std::cout << '\n';
}

void load_from_file(simulation &s, std::string filename) {
  std::ifstream in(filename);
  size_t nbpart;
  in >> nbpart;
  s = simulation(nbpart);
  for (size_t i = 0; i < s.nbpart; ++i) {
    in >> s.mass[i];
    in >> s.x[i] >> s.y[i] >> s.z[i];
    in >> s.vx[i] >> s.vy[i] >> s.vz[i];
    in >> s.fx[i] >> s.fy[i] >> s.fz[i];
  }
  if (!in.good())
    throw "kaboom";
}

void allocate_device(device_simulation &d, size_t nbpart) {
  size_t bytes = nbpart * sizeof(double;
  cudaMalloc((void**)&d.mass, bytes);
  cudaMalloc((void**)&d.x, bytes);
  cudaMalloc((void**)&d.y, bytes);
  cudaMalloc((void**)&d.z, bytes);
  cudaMalloc((void**)&d.vx, bytes);
  cudaMalloc((void**)&d.vy, bytes);
  cudaMalloc((void**)&d.vz, bytes);
  cudaMalloc((void**)&d.fx, bytes);
  cudaMalloc((void**)&d.fy, bytes);
  cudaMalloc((void**)&d.fz, bytes);
}

void free_device(device_simulation &d) {
  cudaFree(d.mass);
  cudaFree(d.x);
  cudaFree(d.y);
  cudaFree(d.z);
  cudaFree(d.vx);
  cudaFree(d.vy);
  cudaFree(d.vz);
  cudaFree(d.fx);
  cudaFree(d.fy);
  cudaFree(d.fz);
}

void copy_to_device(const simulation &s, device_simulation &d) {
  size_t bytes = s.nbpart * sizeof(double);
  cudaMemcpy(d.mass, s.mass.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d.x, s.x.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d.y, s.y.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d.z, s.z.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d.vx, s.vx.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d.vy, s.vy.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d.vz, s.vz.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d.fx, s.fx.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d.fy, s.fy.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d.fz, s.fz.data(), bytes, cudaMemcpyHostToDevice);
}

void copy_from_device(simulation &s, const device_simulation &d) {
  size_t bytes = s.nbpart * sizeof(double);
  cudaMemcpy(s.mass.data(), d.mass, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(s.x.data(), d.x, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(s.y.data(), d.y, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(s.z.data(), d.z, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(s.vx.data(), d.vx, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(s.vy.data(), d.vy, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(s.vz.data(), d.vz, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(s.fx.data(), d.fx, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(s.fy.data(), d.fy, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(s.fz.data(), d.fz, bytes, cudaMemcpyDeviceToHost);
}
int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0] << " <input> <dt> <nbstep> <printevery>"
              << "\n"
              << "input can be:" << "\n"
              << "a number (random initialization)" << "\n"
              << "planet (initialize with solar system)" << "\n"
              << "a filename (load from file in singleline tsv)" << "\n";
    return -1;
  }

  double dt = std::atof(argv[2]); // in seconds
  size_t nbstep = std::atol(argv[3]);
  size_t printevery = std::atol(argv[4]);

  simulation s(1);

  // parse command line
  {
    size_t nbpart = std::atol(argv[1]); // return 0 if not a number
    if (nbpart > 0) {
      s = simulation(nbpart);
      random_init(s);
    } else {
      std::string inputparam = argv[1];
      if (inputparam == "planet") {
        init_solar(s);
      } else {
        load_from_file(s, inputparam);
      }
    }
  }

  for (size_t step = 0; step < nbstep; step++) {
    if (step % printevery == 0)
      dump_state(s);

    reset_force(s);
    for (size_t i = 0; i < s.nbpart; ++i)
      for (size_t j = 0; j < s.nbpart; ++j)
        if (i != j)
          update_force(s, i, j);

    for (size_t i = 0; i < s.nbpart; ++i) {
      apply_force(s, i, dt);
      update_position(s, i, dt);
    }
  }

  // dump_state(s);

  return 0;
}
