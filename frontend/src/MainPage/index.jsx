import React, { useState } from "react";
import Container from "./Container";
import AIResponse from "./Answerbox";
import '../../assets/css/title_section.css';

const App = () => {
    const [query, setQuery] = useState("");
    const [finalquery, setFinalQuery] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    return (
        <div>
            <section className="welcome-hero">
                    <div className="header-text">
                        <h1>監察院案件不二查評估系統</h1>
                        <p className="subheading">
                        </p>
                    </div>
            </section>

            <Container title="輸入查詢內容">
                <div className="input-wrapper">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                />
                <button
                    disabled={isLoading}
                    onClick={async () => {
                        const user_query = query.trim();

                        try {
                            if (user_query === "") {
                                throw Error("沒有輸入問題。");
                            } 

                            setIsLoading(true);
                            setFinalQuery(user_query);
                        } catch (e) {
                            alert(e.message);
                        } finally {
                            setIsLoading(false);
                        }
                    }}
                >
                    輸入訊息
                </button>

                </div>
                
            </Container>

            {finalquery && <AIResponse user_query={finalquery} />}
        </div>
    );
};

export default App;
