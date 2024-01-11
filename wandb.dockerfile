1689835ebde0e71b3d89e7e24f15277d851cd130

FROM python:3.9
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install wandb
COPY s4_debugging_and_logging/exercise_files/wandb_tester.py wandb_tester.py
ENTRYPOINT ["python", "-u", "wandb_tester.py"]
