#pragma once
#include "state.h"
#include <random>

namespace utec::neural_network {

class EnvGym {
  public:
    EnvGym();

    // Reinicia entorno
    State reset();

    // Ejecuta acción y devuelve siguiente estado, reward y done
    State step(int action, float &reward, bool &done);

  private:
    State state_;
    float ball_vx_, ball_vy_;
    float paddle_speed_;

    std::default_random_engine rng_;
    std::uniform_real_distribution<float> dist_pos_;
    std::uniform_real_distribution<float> dist_dir_;

    void update_ball();
    void update_paddle(int action);
    bool check_collision() const;
};

} // namespace utec::neural_network
