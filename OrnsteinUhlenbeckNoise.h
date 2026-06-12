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

    void setSigma(float new_sigma) {
        sigma = new_sigma;
        if (sigma == 0.0f) {
            reset();
        }
    }

    float getSigma() const { return sigma; }

    // Set volatility from a desired *stationary* standard deviation of the walk.
    // An OU process settles to std = sigma/sqrt(2*theta), so invert that. This lets a
    // caller dial "how far the value roams" directly, while theta/dt keep the path
    // smoothness (correlation time) fixed and independent of the chosen amplitude.
    void setStationaryStd(float std) {
        setSigma(std * std::sqrt(2.0f * theta));
    }

    void reset() {
        x = 0.f;
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