import os

MODULES = {
    '1': 'src.main',
    '2': 'src.core.static_capture',
    '3': 'src.core.dynamic_capture',
    '4': 'src.core.gesture_trainer',
    '5': 'src.core.gesture_detector',
    '6': 'src.core.speech_capture',
    '7': 'src.core.speech_trainer'
}

def print_modules():
    print("\nAvailable modules:")
    print("-" * 50)
    for num, module in MODULES.items():
        name = module.split('.')[-1].replace('_', ' ').title()
        print(f"{num}. {name} ({module})")
    print("-" * 50)

def run_module(choice):
    if choice in MODULES:
        command = f"python -m {MODULES[choice]}"
        print(f"\nExecuting: {command}")
        os.system(command)
    else:
        print("Invalid choice!")

def main():
    while True:
        print_modules()
        choice = input("\nEnter the number of the module to run (0 to exit): ")
        
        if choice == '0':
            print("Exiting...")
            break
            
        run_module(choice)
        
        print("\nPress Enter to continue...")
        input()
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    main() 