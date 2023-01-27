import subprocess

subprocess.run("python -mwebbrowser http://localhost:8050/", shell=True)

subprocess.run(
    "docker run  --user root -v bidcache:/home/jovyan/app/data/cache "
    "-p 8050:8050 -e CHOWN_HOME=yes -e CHOWN_HOME_OPTS='-R'  bid-urban-growth",
    shell =True)
