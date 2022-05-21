import numpy as np
from own_env import CustomKoreEnv, build_env, Agent



def main():
    env = build_env()
    model = Agent().build_model(model_parameter="Platzhalter")


    observation = env.reset()
    sum_reward = 0
    done = False
    c = 0
    while (not done) and c < 400:
        observation = np.reshape(observation, [1, 21, 21, 4])
        action_space = model.predict(observation)
        observation_next, reward, done, info = env.step(action_space)
        # TODO hier ggf das Modell trainieren
        observation = observation_next
        sum_reward += reward
        c += 1
    a = 0

if __name__ == "__main__":
    main()