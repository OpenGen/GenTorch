import tracker

state = tracker.State(1, 2, 3, 4, 5)
print(state)

observations = tracker.NextTimeStepObservations(1.123)
print(observations)

extend = tracker.ExtendByOneTimeStep()
print(extend)

constraints = tracker.EmptyConstraints()
print(constraints)

parameters = tracker.Parameters(
    0.005,
    0.005,
    0.01, 0.01,
    0.95, 0.01,
    0.002, 0.01,
    -0.013, 0.01)
print(parameters)
print(parameters.measurement_noise)

rng = tracker.mt19937(1)
print(rng)

model = tracker.Model(0)
print(model)

trace = model.simulate(rng, parameters, False);
print(trace)

pf = tracker.MyParticleSystem(50, rng)

observations = [tracker.NextTimeStepObservations(0.1 * i) for i in range(50)]

pf.init_step(model, parameters, observations[0])
print(pf.effective_sample_size())
pf.resample()

for observation in observations[1:]:
    pf.step(extend, observation)
    print(pf.effective_sample_size())
    pf.resample()

print(pf.traces())
for trace in pf.traces():
    print(trace.state(0))
