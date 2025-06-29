# Bati Bank - Credit Scoring Model

![License](https://img.shields.io/badge/license-MIT-blue.svg)

This project develops an end-to-end credit scoring model for Bati Bank's new buy-now-pay-later (BNPL) service. By analyzing customer transaction data from an eCommerce partner, we build a model that predicts credit risk, assigns a credit score, and helps determine appropriate loan terms for new applicants. The project follows MLOps best practices, including automated data processing, experiment tracking with MLflow, and deployment as a containerized API with a CI/CD pipeline.

## Table of Contents
- [Credit Scoring Business Understanding](#credit-scoring-business-understanding)

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord fundamentally shifts banking regulation from a "one-size-fits-all" approach to one that requires banks to use their own internal estimates of risk for calculating capital requirements. This is known as the Internal Ratings-Based (IRB) approach. This has profound implications for our project:

*   **Auditability and Validation:** Regulators (like the central bank) must be able to audit and validate our credit risk model. They need to understand its logic, assumptions, and limitations. A "black box" model, even if highly accurate, is unacceptable if its decisions cannot be explained. An interpretable model, like Logistic Regression, allows us to explicitly show which factors (e.g., low transaction frequency, certain product categories) contribute to a high-risk score.
*   **Justification of Decisions:** Bati Bank must be able to explain to both regulators and customers why a credit application was denied. This is not just a regulatory requirement but also a matter of fairness and customer trust. An interpretable model provides the reasoning behind each decision.
*   **Model Stability and Governance:** Basel II requires robust model governance. This means our model development process, from data sourcing to final deployment, must be meticulously documented. This ensures the model is stable over time and can be maintained, updated, and re-validated as economic conditions change.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

**Necessity of a Proxy Variable:**
A supervised machine learning model requires a "ground truth" target variable to learn from. In our dataset, we have transaction history but no explicit column that says "this customer defaulted on a loan." Since the BNPL service is new, we have no historical loan performance data.

Therefore, we must create a **proxy variable**—a stand-in for the real-world outcome we want to predict. By analyzing customer behavior, we can make an educated assumption that certain patterns indicate a higher risk of future default. For this project, we will use RFM (Recency, Frequency, Monetary) analysis to identify highly disengaged customers (e.g., those who haven't purchased recently, purchase infrequently, and spend little). We will label this group as our "high-risk" proxy.

**Potential Business Risks:**
This approach is pragmatic but carries significant risks:

*   **Proxy-Target Misalignment:** The core risk is that our proxy (disengaged customer) is not a perfect predictor of actual loan default. A disengaged customer might still be financially stable, while a highly engaged customer could be over-leveraged and a true default risk.
*   **False Positives:** We might incorrectly classify creditworthy customers as "high-risk." This leads to denying them loans, resulting in lost revenue opportunities and a poor customer experience.
*   **False Negatives:** We might incorrectly classify high-risk customers as "low-risk." Approving their loans could lead to actual defaults, causing financial losses for Bati Bank.
*   **Model Bias:** The proxy could introduce unforeseen biases. For example, if certain demographics are naturally less engaged on the platform but are financially responsible, the model could systematically discriminate against them.

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

This trade-off is central to financial modeling.

| Aspect                | Simple Model (Logistic Regression with WoE)                                                                   | Complex Model (Gradient Boosting/XGBoost)                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Interpretability**  | **High.** Coefficients directly show the impact of each feature on the odds of default. Easy to explain to regulators and business stakeholders. | **Low.** It's a "black box." Explaining why one specific customer was denied is very difficult, often requiring complex techniques like SHAP. |
| **Performance**       | **Good.** Often provides solid, robust performance and is less prone to overfitting on noisy data.             | **Excellent.** Typically achieves higher accuracy, precision, and AUC by capturing complex, non-linear relationships in the data. |
| **Regulatory Approval** | **Easier.** Regulators are familiar with and generally trust these models because of their transparency.       | **Harder.** Requires a much higher burden of proof to convince regulators that the model is fair, stable, and not making arbitrary decisions. |
| **Data Requirements** | **More demanding feature engineering.** The model's performance heavily relies on smart feature engineering, such as Weight of Evidence (WoE) and Information Value (IV), to handle non-linearity. | **Less demanding feature engineering.** The algorithm can automatically discover complex interactions, reducing the need for manual feature creation. |
| **Implementation Risk** | **Lower.** Simpler to implement, debug, and maintain.                                                         | **Higher.** More hyperparameters to tune, greater risk of overfitting, and more complex to deploy and monitor. |

**Conclusion for Bati Bank:** In a regulated context, the standard approach is to **start with a simple, interpretable model** like Logistic Regression. We will build this first to serve as our compliant, understandable baseline. Then, we can develop a complex model like Gradient Boosting as a "challenger model" to see how much performance we can gain. The final decision would involve weighing the performance lift against the increased regulatory and maintenance burden.

---
