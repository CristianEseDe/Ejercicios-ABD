// La sección "data" especifica la estructura de los datos.
data {
  int<lower=0> N; // Número de casos en los datos
  int<lower=0> K; // Número de variables predictoras
  matrix[N, K] x; // Matriz de variables predictoras
  int<lower=0, upper=1> y[N]; // Vector de valores de la variable respuesta
}

// Parámetros del modelo.
parameters {
  real alpha;     // Intersección del modelo
  vector[K] beta; // Coeficientes de regresión
}

// Especificación del modelo a estimar. La variable `y` se distribuye según
//   Bernoulli con probabilidad dada por la combinación lineal de las variables
//   `x` y la intersección (`alpha`).
model {
  beta  ~ normal(0, 2);
  alpha ~ normal(0, 2);
  y ~ bernoulli_logit(alpha + x * beta);
}

generated quantities {
  real alpha_squared;
  vector[K] beta_squared;

  alpha_squared = alpha * alpha;
  for (k in 1:K) {
    beta_squared[k] = beta[k] * beta[k];
  }

  // También añado log-likelihood y predicciones
  vector[N] log_lik;
  vector[N] y_pred;

  for (n in 1:N) {
    real p = inv_logit(alpha + x[n] * beta);
    log_lik[n] = bernoulli_lpmf(y[n] | p);
    y_pred[n] = bernoulli_rng(p);
  }
}
