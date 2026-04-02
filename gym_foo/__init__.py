from gymnasium.envs.registration import register

register(
    id='foo-v1',
    entry_point='gym_foo.envs:FooEnv',
)