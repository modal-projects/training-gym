## TODO
Tasks that are currently not fully validated.
- Add defaults to ensure each framework can be set up with <10 lines of code.
- Examples missing:
    - slime with LoRA
    - non-cluster examples with a smaller model to ensure cost savings.
    - using it on a larger model (e.g. GLM 4.7)
- In logs, tell user we are converting checkpoint from mt to hf. Do this in serve.
- Remove gym-server from the modal-training-gym client side and move to shared functions that operate on volumes. Still keep the gym-server for the dashboard but when training results finish they just call shared fns to write.
- How do u add tool calling to your model: eval is single turn in how it works right now