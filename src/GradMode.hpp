#ifndef GRAD_MODE_HPP
#define GRAD_MODE_HPP

#include <stack>
#include <mutex>

class GradMode {
    public:
        // Get current grad state
        static bool isEnabled(){
            std::lock_guard<std::mutex> lock(mutex_);
            return !grad_states_.empty() && grad_states_.top();
        }

        // Push new gradient state
        static void push(bool enabled){
            std::lock_guard<std::mutex> lock(mutex_);
            grad_states_.push(enabled);
        }

        // Pop current gradient state
        static void pop(){
            std::lock_guard<std::mutex> lock(mutex_);
            if(!grad_states_.empty()){
                grad_states_.pop();
            }
        }

        // Set gradient mode without stack manipulation
        static void set(bool enabled){
            std::lock_guard<std::mutex> lock(mutex_);
            if(grad_states_.empty()){
                grad_states_.push(enabled);
            }
            else{
                grad_states_.top() = enabled;
            }
        }

    private:
        static std::stack<bool> grad_states_;
        static std::mutex mutex_;
    
};

class GradientGuard {
    public:
        explicit GradientGuard(bool enabled){
            GradMode::push(enabled);
        }
        ~GradientGuard(){
            GradMode::pop();
        }
};

class NoGradGuard{
    public:
        NoGradGuard() : guard_(false){}
    private:
        GradientGuard guard_;
};

class EnableGradGuard{
    public:
        EnableGradGuard() : guard_(true){}
    private:
        GradientGuard guard_;
};

#endif