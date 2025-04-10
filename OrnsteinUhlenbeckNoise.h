#include <random>

class OrnsteinUhlenbeckNoise {
public:
    OrnsteinUhlenbeckNoise(float theta, float mu, float sigma, float dt, float initial = 0.0)
        : theta(theta), mu(mu), sigma(sigma), dt(dt), x(initial),
          generator(std::random_device{}()), distribution(0.0, 1.0) {}

    float sample() {
        float dW = distribution(generator) * std::sqrt(dt); // Wiener process increment
        x += theta * (mu - x) * dt + sigma * dW;
        return x;
    }

private:
    float theta;   // Mean reversion speed
    float mu;      // Long-term mean
    float sigma;   // Volatility (noise intensity)
    float dt;      // Time step
    float x;       // Current state

    std::mt19937 generator;
    std::normal_distribution<float> distribution;
};