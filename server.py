from openreward.environments import Server

from npp_sim import NuclearPlantEnvironment

if __name__ == "__main__":
    server = Server([NuclearPlantEnvironment])
    server.run()
