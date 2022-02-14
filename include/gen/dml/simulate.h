#ifndef GENTORCH_SIMULATE_H
#define GENTORCH_SIMULATE_H

namespace gen::dml {

template<typename RNG, typename Model>
class DMLSimulateTracer {
public:
    typedef typename Model::args_type args_type;
    typedef typename Model::parameters_type parameters_type;
    using trace_type = DMLTrace<Model>;

    explicit DMLSimulateTracer(RNG& rng,
                               Model gen_fn_with_args,
                               parameters_type &parameters,
                               bool prepare_for_gradients, bool assert_retval_grad) :
            finished_(false),
            rng_{rng},
            trace_{std::make_unique<trace_type>(std::move(gen_fn_with_args), prepare_for_gradients,
                                                assert_retval_grad, parameters)},
            prepare_for_gradients_{prepare_for_gradients},
            parameters_{parameters} {
        assert(!(prepare_for_gradients && c10::InferenceMode::is_enabled()));
    }

    const args_type& get_args() const { return trace_->get_args(); }

    template<typename CalleeType, typename CalleeParametersType>
    typename CalleeType::return_type
    call(Address &&address, CalleeType &&gen_fn_with_args, CalleeParametersType &parameters) {
        typedef typename CalleeType::args_type callee_args_type;
        typedef typename CalleeType::trace_type callee_trace_type;
        typedef typename CalleeType::return_type callee_return_type;
        assert(!finished_); // if this assertion fails, it is a bug in DML not user code
        callee_trace_type &subtrace = trace_->add_subtrace(address,
                                                           gen_fn_with_args.simulate(
                                                                   rng_, parameters,
                                                                   SimulateOptions().precompute_gradient(
                                                                           prepare_for_gradients_)));
        const auto &value = subtrace.return_value();
        if (prepare_for_gradients_) {
            return trace_->make_tracked_return_value(subtrace, gen_fn_with_args.get_args(), value);
        } else {
            return value; // copy the value
        }
    }

    template<typename CalleeType>
    typename CalleeType::return_type
    call(Address&& address, CalleeType&& gen_fn_with_args) {
        return call(std::move(address), std::forward<CalleeType>(gen_fn_with_args), gen::empty_module_singleton);
    }

    std::unique_ptr <DMLTrace<Model>> finish(typename Model::return_type value) {
        assert(!(prepare_for_gradients_ && c10::InferenceMode::is_enabled()));
        finished_ = true;
        trace_->set_value(value);
        return std::move(trace_);
    }

    // for use in the body of the exec() function
    parameters_type &get_parameters() { return parameters_; }

    bool prepare_for_gradients() const { return prepare_for_gradients_; }

private:
    bool finished_;
    RNG& rng_;
    std::unique_ptr<DMLTrace<Model>> trace_;
    bool prepare_for_gradients_;
    parameters_type &parameters_;
};

template<typename Model, typename Args, typename Return, typename Parameters>
template<typename RNG>
std::unique_ptr<DMLTrace<Model>> DMLGenFn<Model, Args, Return, Parameters>::simulate(RNG& rng, Parameters &parameters,
                                                                                      const SimulateOptions &options) {
    c10::InferenceMode guard{
            !options.precompute_gradient()}; // inference mode is on if we are not preparing for gradients
    DMLSimulateTracer<RNG, Model> tracer{
        rng, *static_cast<Model*>(this), parameters, options.precompute_gradient(), assert_retval_grad_};
    auto value = static_cast<Model*>(this)->forward(tracer);
    return tracer.finish(value);
}

}


#endif //GENTORCH_SIMULATE_H
