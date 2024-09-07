from tqdm import tqdm  # Add this line to import tqdm

def train_vit(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", desc=f"Epoch {epoch+1}/{epochs}")  # tqdm progress bar
        for batch in tepoch:
            inputs, targets = batch
            inputs, targets = inputs.float(), targets.float()  # Convert to float

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Reshape outputs to match target dimensions
            outputs = outputs.view(-1, 1, 10, 10)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tepoch.set_postfix(loss=total_loss / len(dataloader))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")
