#include "GradMode.hpp"

std::stack<bool> GradMode::grad_states_({true});
std::mutex GradMode::mutex_;