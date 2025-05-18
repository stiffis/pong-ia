#include "EnvGym.h"

namespace utec::neural_network {

EnvGym::EnvGym()
    : rng_(std::random_device{}()), dist_pos_(0.2f, 0.8f),
      dist_dir_(-0.03f, 0.03f), paddle_speed_(0.04f) {}

State EnvGym::reset() {
    // Inicializamos posiciones de bola y paleta
    state_.ball_x = 0.5f;
    state_.ball_y = dist_pos_(rng_);
    state_.paddle_y = 0.5f;

    // Velocidad inicial bola hacia la paleta (a la derecha)
    ball_vx_ = 0.03f;
    ball_vy_ = dist_dir_(rng_);
    return state_;
}

void EnvGym::update_ball() {
    state_.ball_x += ball_vx_;
    state_.ball_y += ball_vy_;

    // Rebote en paredes superior e inferior
    if (state_.ball_y < 0.f) {
        state_.ball_y = 0.f;
        ball_vy_ = -ball_vy_;
    } else if (state_.ball_y > 1.f) {
        state_.ball_y = 1.f;
        ball_vy_ = -ball_vy_;
    }
}

void EnvGym::update_paddle(int action) {
    // Acción: -1,0,1 mueve paleta arriba, quieto o abajo
    state_.paddle_y += paddle_speed_ * action;

    if (state_.paddle_y < 0.f)
        state_.paddle_y = 0.f;
    if (state_.paddle_y > 1.f)
        state_.paddle_y = 1.f;
}

bool EnvGym::check_collision() const {
    // Consideramos que la paleta cubre un rango vertical centrado en paddle_y
    // y la bola se "devuelve" si está dentro de ese rango cuando la bola llega
    // a la paleta_x
    constexpr float paddle_x = 0.95f; // posición horizontal paleta
    constexpr float paddle_height = 0.2f;

    if (state_.ball_x >= paddle_x) {
        float diff = state_.ball_y - state_.paddle_y;
        return (diff >= -paddle_height / 2 && diff <= paddle_height / 2);
    }
    return false;
}

State EnvGym::step(int action, float &reward, bool &done) {
    update_paddle(action);
    update_ball();

    reward = 0.f;
    done = false;

    constexpr float paddle_x = 0.95f;

    if (state_.ball_x >= paddle_x) {
        if (check_collision()) {
            // Devuelve bola: invertimos velocidad horizontal con variación
            // vertical
            ball_vx_ = -ball_vx_;
            ball_vy_ = dist_dir_(rng_);
            reward = 1.f;
        } else {
            // Falló el punto
            reward = -1.f;
            done = true;
        }
    } else if (state_.ball_x < 0.f) {
        // Bola salió por izquierda (reiniciar o puntuar)
        ball_vx_ = -ball_vx_;
        reward = 0.f;
    }

    return state_;
}

} // namespace utec::neural_network
