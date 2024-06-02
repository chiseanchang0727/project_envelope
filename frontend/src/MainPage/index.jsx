import React, { useState } from "react";
import Container from "./Container";
import AIResponse from "./Answerbox";

const App = () => {
    const [query, setQuery] = useState("");
    const [finalquery, setFinalQuery] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const handleButtonClick = async () => {
        const user_query = query.trim();

        try {
            if (user_query === "") {
                throw Error("沒有問題輸入.");
            }

            setIsLoading(true);
            setFinalQuery(user_query);
        } catch (e) {
            alert(e.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div>
            <section className="welcome-hero">
                <div className="container">
                    <div className="header-text">
                        <h1>資料查詢AI</h1>
                        <p className="subheading">
                            透過對AI詢問在廣大的資料庫中尋找相關訊息
                        </p>
                    </div>
                </div>
            </section>

            <Container title="輸入問題">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                />
                <button
                    disabled={isLoading}
                    onClick={handleButtonClick}
                >
                    輸入訊息
                </button>
            </Container>

            <AIResponse user_query={finalquery} />
        </div>
    );
};

export default App;
