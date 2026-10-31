// ====================================
// PAGE NAVIGATION
// ====================================

function showPage(pageName) {
    document.querySelectorAll(".page").forEach(page => {
        page.classList.remove("active-page");
    });

    const selectedPage = document.getElementById(pageName);
    if (selectedPage) {
        selectedPage.classList.add("active-page");
    }

    document.querySelectorAll(".nav-links a").forEach(link => {
        link.classList.toggle(
            "active",
            link.dataset.page === pageName
        );
    });

    const navLinks = document.getElementById("navLinks");
    if (navLinks) {
        navLinks.classList.remove("active");
    }

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}

// Navigation links
document.querySelectorAll("[data-page]").forEach(link => {
    link.addEventListener("click", function(event) {
        event.preventDefault();
        showPage(this.dataset.page);
    });
});

// Mobile menu toggle
const menuToggle = document.getElementById("menuToggle");
if (menuToggle) {
    menuToggle.addEventListener("click", () => {
        document.getElementById("navLinks").classList.toggle("active");
    });
}

// ====================================
// SAMPLE DATA & RESET HELPERS
// ====================================

const SAMPLE_HOUSING_DATA = {
    CRIM: 0.02729,
    ZN: 18.0,
    INDUS: 7.07,
    CHAS: "0",
    NOX: 0.469,
    RM: 7.185,
    AGE: 61.1,
    DIS: 4.9671,
    RAD: 2,
    TAX: 242,
    PTRATIO: 17.8,
    B: 392.83,
    LSTAT: 4.03
};

const sampleBtn = document.getElementById("sampleBtn");
if (sampleBtn) {
    sampleBtn.addEventListener("click", () => {
        Object.entries(SAMPLE_HOUSING_DATA).forEach(([key, val]) => {
            const input = document.getElementById(key);
            if (input) {
                input.value = val;
                input.dispatchEvent(new Event("change"));
            }
        });
        const errorBox = document.getElementById("errorBox");
        if (errorBox) errorBox.classList.add("hidden");
    });
}

const clearBtn = document.getElementById("clearBtn");
if (clearBtn) {
    clearBtn.addEventListener("click", () => {
        const form = document.getElementById("predictionForm");
        if (form) form.reset();
        const resultCard = document.getElementById("resultCard");
        if (resultCard) resultCard.classList.add("hidden");
        const errorBox = document.getElementById("errorBox");
        if (errorBox) errorBox.classList.add("hidden");
    });
}

// ====================================
// HOUSING PRICE PREDICTION
// ====================================

const form = document.getElementById("predictionForm");

if (form) {
    form.addEventListener("submit", async function(event) {
        event.preventDefault();

        const loading = document.getElementById("loading");
        const resultCard = document.getElementById("resultCard");
        const errorBox = document.getElementById("errorBox");
        const button = document.getElementById("predictButton");

        if (resultCard) resultCard.classList.add("hidden");
        if (errorBox) errorBox.classList.add("hidden");

        if (loading) loading.classList.remove("hidden");
        if (button) {
            button.disabled = true;
            button.innerHTML = "<span>Analyzing Features...</span>";
        }

        try {
            const formData = new FormData(form);
            const data = {};

            formData.forEach((value, key) => {
                data[key] = parseFloat(value);
            });

            // Smart API routing: works when opened from Flask, localhost, or file://
            let targetUrl = "/predict";
            if (window.location.protocol === "file:" || !window.location.port) {
                targetUrl = "http://127.0.0.1:5000/predict";
            } else if (window.location.port && window.location.port !== "5000") {
                targetUrl = "http://127.0.0.1:5000/predict";
            }

            let response;
            try {
                response = await fetch(targetUrl, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });
            } catch (networkErr) {
                // If local relative route failed, try direct 127.0.0.1:5000
                if (targetUrl !== "http://127.0.0.1:5000/predict") {
                    response = await fetch("http://127.0.0.1:5000/predict", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(data)
                    });
                } else {
                    throw new Error("Unable to connect to Flask server. Please make sure `python app.py` is running on port 5000.");
                }
            }

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(
                    result.error || (result.missing_features ? "Missing features: " + result.missing_features.join(", ") : "Prediction failed.")
                );
            }

            // Extract numeric prediction
            const priceThousands = (result.price_thousands !== undefined) 
                ? Number(result.price_thousands) 
                : Number(result.prediction);

            const priceDollars = (result.price !== undefined) 
                ? Number(result.price) 
                : (priceThousands * 1000);

            // Format prices cleanly
            const formattedPrice = new Intl.NumberFormat("en-US", {
                style: "currency",
                currency: "USD",
                maximumFractionDigits: 0
            }).format(priceDollars);

            const resultPriceEl = document.getElementById("resultPrice");
            if (resultPriceEl) {
                resultPriceEl.textContent = formattedPrice;
            }

            const resultThousandsEl = document.getElementById("resultThousands");
            if (resultThousandsEl) {
                resultThousandsEl.textContent = `Median estimated value: $${priceThousands.toFixed(2)}k USD`;
            }

            const resultModelEl = document.getElementById("resultModel");
            if (resultModelEl) {
                resultModelEl.textContent = result.best_algorithm || "Ensemble Regressor";
            }

            const resultScoreEl = document.getElementById("resultScore");
            if (resultScoreEl) {
                if (result.r2_score !== undefined) {
                    const pct = (result.r2_score * 100).toFixed(1);
                    resultScoreEl.textContent = `${pct}% (R² = ${result.r2_score})`;
                } else {
                    resultScoreEl.textContent = "High Precision";
                }
            }

            const resultRangeEl = document.getElementById("resultRange");
            if (resultRangeEl) {
                const low = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(priceDollars * 0.95);
                const high = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(priceDollars * 1.05);
                resultRangeEl.textContent = `${low} – ${high}`;
            }

            if (resultCard) {
                resultCard.classList.remove("hidden");
                resultCard.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });
            }

        } catch (error) {
            if (errorBox) {
                errorBox.innerHTML = `<strong>Prediction Error:</strong> ${error.message}`;
                errorBox.classList.remove("hidden");
                errorBox.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });
            }
        } finally {
            if (loading) loading.classList.add("hidden");
            if (button) {
                button.disabled = false;
                button.innerHTML = "Calculate House Price →";
            }
        }
    });
}
