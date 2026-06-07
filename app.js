const form = document.getElementById('predict-form');
const result = document.getElementById('result');
const nameInput = document.getElementById('name');

function setResult(message, isError = false) {
  result.innerHTML = message;
  result.classList.toggle('muted', !isError);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const name = nameInput.value.trim();
  if (!name) {
    setResult('Please enter a name.', true);
    return;
  }

  setResult('Predicting...');

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name }),
    });

    const data = await response.json();

    if (!response.ok) {
      setResult(data.error || 'Prediction failed.', true);
      return;
    }

    setResult(
      `<strong>${data.name}</strong> is predicted as <strong>${data.predicted_gender}</strong> with confidence <strong>${data.confidence}</strong>.`
    );
  } catch (error) {
    setResult('Could not reach the prediction API.', true);
  }
});
