class ConvSBERT(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvSBERT, self).__init__()
        self.num_classes = num_classes
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(64, num_classes)  # output layer with 'num_classes' units

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2 = inputs
        # Convolutional
        x1 = self.conv1(x1.unsqueeze(2))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = self.conv1(x2.unsqueeze(2))  # Add a channel dimension
        x2 = self.maxpool(x2)
        # TanH
        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)
        # Reshape
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        # Calculate difference
        concatenated = x2 - x1
        # Feed to forward composition layers
        x = self.fc1(concatenated)
        x = self.dropout1(x)
        final_layer_output = self.fc2(x)  # Linear output, no activation

        return final_layer_output

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
        # Clean data
        training_data["sentence1_embeddings"] = model.encode([s.strip() for s in training_data["sentence1"]])
        training_data["sentence2_embeddings"] = model.encode([s.strip() for s in training_data["sentence2"]])
        validation_data["sentence1_embeddings"] = model.encode([s.strip() for s in validation_data["sentence1"]])
        validation_data["sentence2_embeddings"] = model.encode([s.strip() for s in validation_data["sentence2"]])
        # Stack embeddings for batch processing
        sentence1_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)
        sentence1_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence1_embeddings"]), dim=0
        )
        sentence2_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence2_embeddings"]), dim=0
        )
        print(sentence2_training_embeddings.shape)

        device = torch.device(device)
        self.to(device)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []

            for i in range(0, len(training_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i: i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i: i + batch_size]

                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(training_data["label"].iloc[i: i + batch_size].values,
                                                          dtype=torch.long).to(device)

                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)

                # Forward pass
                outputs: torch.Tensor = model([s1_embedding, s2_embedding])

                # Compute the loss
                loss: float = criterion(outputs, batch_labels)

                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()

                # Optimize (update model parameters)
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())

            average_loss: float = running_loss / (len(training_data) / batch_size)

            # Calculate training accuracy and F1-score
            training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
            training_f1: float = f1_score(all_true_labels, all_predicted_labels,
                                          average='macro')  # You can choose 'micro' or 'weighted' as well

            train_accuracy_values.append(training_accuracy)
            train_f1_values.append(training_f1)

            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i: i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i: i + batch_size]

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)

                    # Forward pass for validation
                    val_outputs: torch.Tensor = model([s1_embedding, s2_embedding])

                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)

                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_true_labels = validation_data["label"].values
            val_accuracy: float = accuracy_score(val_true_labels, all_val_predicted_labels)
            val_f1: float = f1_score(val_true_labels, all_val_predicted_labels,
                                     average='macro')  # You can choose 'micro' or 'weighted' as well

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}")
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(embedding_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(embedding_to_tensor)

        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i: i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i: i + batch_size].to(device)
                true_labels: torch.Tensor = torch.tensor(test_data["label"].iloc[i: i + batch_size].values,
                                                         dtype=torch.long).to(device)

                # Forward pass for predictions
                output: torch.Tensor = model([s1_embedding, s2_embedding])
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)

                # Calculate accuracy and F1-score for this batch
                true_labels_cpu: np.ndarray = true_labels.cpu().numpy()
                predicted_classes_cpu: np.ndarray = predicted_classes.cpu().numpy()
                all_true_labels.extend(true_labels_cpu)
                final_predictions = np.append(final_predictions, predicted_classes_cpu)
            return final_predictions


class ConvSBERTTri(nn.Module):
    def __init__(self):
        super(ConvSBERTTri, self).__init__()
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.shared_conv = nn.Conv1d(in_channels=768 * 2, out_channels=256, kernel_size=1)  # Shared Convolutional Layer

    def forward(self, inputs: tuple) -> torch.Tensor:
        i1, i2 = inputs
        # Convolutional
        x1 = F.relu(self.conv1(i1.unsqueeze(2)))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = F.relu(self.conv1(i2.unsqueeze(2)))  # Add a channel dimension
        x2 = self.maxpool(x2)
        # TanH
        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)
        # Reshape
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        # Calculate difference
        concatenated = x2 - x1
        # Combine x1 and x2 and pass through the shared convolutional layer
        x3 = torch.cat((i1, i2), dim=1)
        x3 = F.relu(self.shared_conv(x3.unsqueeze(2)))
        x3 = self.maxpool(x3)
        x3 = torch.tanh(x3)
        x3 = x3.view(x3.size(0), -1)
        # x = torch.cat((x2, concatenated), dim=1)
        # x = concatenated + x3
        # x = concatenated - x3
        # x = x3 - concatenated
        # Feed to forward composition layers
        x = self.fc1(x3)
        x = self.dropout1(x)
        final_layer_output = torch.sigmoid(self.fc2(x))
        return final_layer_output

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(embedding_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(embedding_to_tensor)
        # Stack embeddings for batch processing
        sentence1_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)
        sentence1_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence1_embeddings"]), dim=0
        )
        sentence2_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence2_embeddings"]), dim=0
        )

        device = torch.device(device)
        self.to(device)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_predicted_labels: list = []
            all_true_labels: list = []
            for i in range(0, len(training_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i : i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i : i + batch_size]
                # Get the corresponding labels for this batch
                batch_labels: np.ndarray = training_data["label"].iloc[i : i + batch_size].values
                batch_labels: torch.Tensor = (
                    torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                )

                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)
                # Forward pass
                # outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations, batch_s1_feature_a, batch_s1_feature_b, batch_s2_feature_a, batch_s2_feature_b])
                outputs: torch.Tensor = model([s1_embedding, s2_embedding])
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to binary predictions (0 or 1)
                predicted_labels: np.ndarray = (outputs >= 0.5).float().view(-1).cpu().numpy()
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()
                all_predicted_labels.extend(predicted_labels)
                all_true_labels.extend(true_labels)
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels)

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: np.ndarray = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size]

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)

                    # Forward pass for validation
                    val_outputs: torch.Tensor = model([s1_embedding, s2_embedding])

                    # Convert validation outputs to binary predictions (0 or 1)
                    val_predicted_labels: np.ndarray = (val_outputs >= 0.5).float().view(-1).cpu().numpy()
                    all_val_predicted_labels.extend(val_predicted_labels)

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels)
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(embedding_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(embedding_to_tensor)
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                output: torch.Tensor = model([s1_embedding, s2_embedding])
                predicted_labels: np.ndarray = (output >= 0.5).float().cpu().numpy()
                final_predictions = np.append(final_predictions, predicted_labels)
            return final_predictions


class ConvSBERTNeg(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvSBERTNeg, self).__init__()
        self.num_classes = num_classes
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(65, num_classes)  # output layer with 'num_classes' units

    def forward(self, inputs: tuple):
        x1, x2, num_negation = inputs

        x1 = F.relu(self.conv1(x1.unsqueeze(2)))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = F.relu(self.conv1(x2.unsqueeze(2)))  # Add a channel dimension
        x2 = self.maxpool(x2)

        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        # Subtract the representations of x1 and x2
        concatenated = x2 - x1
        x = self.fc1(concatenated)
        x = self.dropout1(x)

        # Add num_negation to x
        x = torch.cat((x, num_negation), dim=1)

        final_layer_output = self.fc2(x)

        return final_layer_output

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(embedding_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(embedding_to_tensor)
        # Stack embeddings for batch processing
        sentence1_training_embeddings = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)
        sentence1_validation_embeddings = torch.stack(list(validation_data["sentence1_embeddings"]), dim=0)
        sentence2_validation_embeddings = torch.stack(list(validation_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_predicted_labels: list = []
            all_true_labels: list = []
            for i in range(0, len(training_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i : i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i : i + batch_size]
                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(training_data["label"].iloc[i: i + batch_size].values,
                                                          dtype=torch.long).to(device)
                # Get additional feature values
                batch_labels: torch.Tensor = torch.tensor(training_data["negation"].iloc[i: i + batch_size].values,
                                                          dtype=torch.long).to(device)
                batch_negations: np.ndarray = torch.tensor(training_data["negation"].iloc[i : i + batch_size].values,
                                                          dtype=torch.long).view(-1, 1).to(device)
                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)
                # Forward pass
                outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels, average='macro')

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size]

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)
                    batch_negations: np.ndarray = torch.tensor(validation_data["negation"].iloc[i: i + batch_size].values,
                                                               dtype=torch.long).view(-1, 1).to(device)

                    # Forward pass for validation
                    val_outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])
                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)
                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_true_labels = validation_data["label"].values
            val_accuracy: float = accuracy_score(val_true_labels, all_val_predicted_labels)
            val_f1: float = f1_score(val_true_labels, all_val_predicted_labels,
                                     average='macro')  # You can choose 'micro' or 'weighted' as well

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(embedding_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(embedding_to_tensor)
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []
            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                # Get additional feature values
                true_labels: torch.Tensor = torch.tensor(test_data["label"].iloc[i: i + batch_size].values,
                                                         dtype=torch.long).to(device)
                num_negations: np.ndarray = test_data["negation"].iloc[i: i + batch_size].values
                batch_negations: torch.Tensor = (
                    torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                )

                output: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                all_true_labels.extend(true_labels.cpu().numpy())
                final_predictions = np.append(final_predictions, predicted_classes.cpu().numpy())
            return final_predictions


# Add PH
class ConvSBERTPH(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvSBERTPH, self).__init__()
        # Define model layers
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(1186, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(64, self.num_classes)  # output layer

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2, ph1a, ph1b, ph2a, ph2b = inputs  # ph shape: [x, 260, 3] amd [x, 50, 3]
        # Convolutional
        x1 = F.relu(self.conv1(x1.unsqueeze(2)))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = F.relu(self.conv1(x2.unsqueeze(2)))  # Add a channel dimension
        x2 = self.maxpool(x2)
        # TanH
        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)
        # Reshape
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        # Calculate difference
        x_concat = x2 - x1  # [x, 256]
        pha_concat = ph1a - ph2a  # [x, 260, 3]
        phb_concat = ph1b + ph2b  # [x, 50, 3]

        pha_concat_reshaped = pha_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 260*3]
        phb_concat_reshaped = phb_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 50*3]

        # Now you can concatenate x_concat, pha_concat_reshaped, and phb_concat_reshaped along dimension 1
        final_input = torch.cat(
            (x_concat, pha_concat_reshaped, phb_concat_reshaped), dim=1
        )  # [x, 256 + 260*3 + 50*3] ([x, 1186])

        # Feed to forward composition layers
        x = self.fc1(final_input)
        x = self.dropout1(x)
        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(embedding_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(embedding_to_tensor)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            training_data[column] = training_data[column].apply(ph_to_tensor)
            # Validation cleaning
            validation_data[column] = validation_data[column].apply(ph_to_tensor)

        # Stack embeddings for batch processing
        sentence1_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)
        sentence1_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence1_embeddings"]), dim=0
        )
        sentence2_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence2_embeddings"]), dim=0
        )

        device = torch.device(device)
        self.to(device)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_predicted_labels: list = []
            all_true_labels: list = []
            for i in range(0, len(training_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = torch.Tensor(sentence1_training_embeddings[i : i + batch_size]).to(device)
                s2_embedding: torch.Tensor = torch.Tensor(sentence2_training_embeddings[i : i + batch_size]).to(device)
                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(training_data["label"].iloc[i: i + batch_size].values,
                                                          dtype=torch.long).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)

                # Forward pass
                outputs: torch.Tensor = model(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels, average='macro')

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size].to(device)
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size].to(device)

                    # Prepare PH vectors
                    batch_s1_feature_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s1_feature_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)

                    # Forward pass
                    val_outputs: torch.Tensor = model(
                        [
                            s1_embedding,
                            s2_embedding,
                            batch_s1_feature_a,
                            batch_s1_feature_b,
                            batch_s2_feature_a,
                            batch_s2_feature_b,
                        ]
                    )

                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)

                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels, average='macro')
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(embedding_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(embedding_to_tensor)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Test cleaning
            test_data[column] = test_data[column].apply(ph_to_tensor)

        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i: i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i: i + batch_size].to(device)
                true_labels: torch.Tensor = torch.tensor(test_data["label"].iloc[i: i + batch_size].values,
                                                         dtype=torch.long).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                output: torch.Tensor = model(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)

                # Calculate accuracy and F1-score for this batch
                true_labels_cpu: np.ndarray = true_labels.cpu().numpy()
                predicted_classes_cpu: np.ndarray = predicted_classes.cpu().numpy()
                all_true_labels.extend(true_labels_cpu)
                final_predictions = np.append(final_predictions, predicted_classes_cpu)
            return final_predictions


# Add PH
class ConvSBERTNegPH(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvSBERTNegPH, self).__init__()
        # Define model layers
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(1186, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(65, self.num_classes)  # output layer

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2, num_negation, ph1a, ph1b, ph2a, ph2b = inputs  # ph shape: [x, 260, 3] amd [x, 50, 3]
        # Convolutional
        x1 = F.relu(self.conv1(x1.unsqueeze(2)))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = F.relu(self.conv1(x2.unsqueeze(2)))  # Add a channel dimension
        x2 = self.maxpool(x2)
        # TanH
        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)
        # Reshape
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        # Calculate difference
        x_concat = x2 - x1  # [x, 256]
        pha_concat = ph1a - ph2a  # [x, 260, 3]
        phb_concat = ph1b + ph2b  # [x, 50, 3]

        pha_concat_reshaped = pha_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 260*3]
        phb_concat_reshaped = phb_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 50*3]

        # Now you can concatenate x_concat, pha_concat_reshaped, and phb_concat_reshaped along dimension 1
        final_input = torch.cat(
            (x_concat, pha_concat_reshaped, phb_concat_reshaped), dim=1
        )  # [x, 256 + 260*3 + 50*3] ([x, 1186])
        # Feed to forward composition layers
        x = self.fc1(final_input)
        x = self.dropout1(x)
        # Add num_negation to x
        x = torch.cat((x, num_negation), dim=1)
        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(embedding_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(embedding_to_tensor)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            training_data[column] = training_data[column].apply(ph_to_tensor)
            # Validation cleaning
            validation_data[column] = validation_data[column].apply(ph_to_tensor)

        # Stack embeddings for batch processing
        sentence1_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)
        sentence1_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence1_embeddings"]), dim=0
        )
        sentence2_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence2_embeddings"]), dim=0
        )

        device = torch.device(device)
        self.to(device)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_predicted_labels: list = []
            all_true_labels: list = []
            for i in range(0, len(training_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = torch.Tensor(sentence1_training_embeddings[i : i + batch_size]).to(device)
                s2_embedding: torch.Tensor = torch.Tensor(sentence2_training_embeddings[i : i + batch_size]).to(device)
                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(training_data["label"].iloc[i: i + batch_size].values,
                                                          dtype=torch.long).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)

                batch_negations: np.ndarray = torch.tensor(training_data["negation"].iloc[i : i + batch_size].values,
                                                          dtype=torch.long).view(-1, 1).to(device)

                # Forward pass
                outputs: torch.Tensor = model(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_negations,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels, average='macro')

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size].to(device)
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size].to(device)
                    batch_negations: np.ndarray = torch.tensor(validation_data["negation"].iloc[i: i + batch_size].values,
                                                               dtype=torch.long).view(-1, 1).to(device)

                    # Prepare PH vectors
                    batch_s1_feature_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s1_feature_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)

                    # Forward pass
                    val_outputs: torch.Tensor = model(
                        [
                            s1_embedding,
                            s2_embedding,
                            batch_negations,
                            batch_s1_feature_a,
                            batch_s1_feature_b,
                            batch_s2_feature_a,
                            batch_s2_feature_b,
                        ]
                    )

                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)

                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels, average='macro')
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(embedding_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(embedding_to_tensor)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Test cleaning
            test_data[column] = test_data[column].apply(ph_to_tensor)

        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i: i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i: i + batch_size].to(device)
                batch_negations: np.ndarray = torch.tensor(test_data["negation"].iloc[i : i + batch_size].values,
                                                          dtype=torch.long).view(-1, 1).to(device)
                true_labels: torch.Tensor = torch.tensor(test_data["label"].iloc[i: i + batch_size].values,
                                                         dtype=torch.long).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                output: torch.Tensor = model(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_negations,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)

                # Calculate accuracy and F1-score for this batch
                true_labels_cpu: np.ndarray = true_labels.cpu().numpy()
                predicted_classes_cpu: np.ndarray = predicted_classes.cpu().numpy()
                all_true_labels.extend(true_labels_cpu)
                final_predictions = np.append(final_predictions, predicted_classes_cpu)
            return final_predictions