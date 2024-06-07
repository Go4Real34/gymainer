import os
import subprocess

def main():
    current_os = os.name
    
    cwd = os.getcwd()
    if os.name == "posix":
        venv_python = os.path.join(cwd, "venv", "bin", "python")
    
    elif os.name == "nt":
        venv_python = os.path.join(cwd, "venv", "Scripts", "python")
        
    option = input("Do you want to train or test the model (TRAIN / TEST / GAME)?: ")
    if option.upper() == "TRAIN":
        subprocess.run([venv_python, "model_train.py"])
    
    elif option.upper() == "TEST":
        subprocess.run([venv_python, "model_test.py"])
    
    elif option.upper() == "GAME":
        subprocess.run([venv_python, "model_game.py"])
    
    else:
        print("Invalid option!")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
    