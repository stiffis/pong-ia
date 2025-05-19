#ifndef STATE_H
#define STATE_H

namespace utec::neural_network {

struct State {
    float ball_x, ball_y; // posición de la bola normalizada [0,1]
    float paddle_y;       // posición vertical paleta normalizada [0,1]

    // Constructor para facilidad
    State(float bx = 0.f, float by = 0.f, float py = 0.f)
        : ball_x(bx), ball_y(by), paddle_y(py) {}
};

} // namespace utec::neural_network
#endif // !STATE_H
