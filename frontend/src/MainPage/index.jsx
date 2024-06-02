import React, { useState } from "react";
import Container from "./Container";


const App = () => {
    const [query, setQuery] = useState("");
    const [finalquery, setFinalQuery] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    return (
        <div>
            <section class="welcome-hero">
            <div class="container">
                <div class="header-text">
                <h1>資料查詢AI</h1>
                <p class="subheading">
                    透過對AI詢問在廣大的資料庫中尋找相關訊息
                </p>
                </div>
            </div>
        </section>

        <Container title = "輸入問題">
            <input 
                type="string"
                value={query}
                onChange={(e) => {
                    setQuery(e.target.value);
                }}
            />
            <button 
                disabled={isLoading}>
                onClick={async (e) => {
                    const user_query = query.trim();

                    try {
                        if (user_query === ""){
                            throw Error("沒有問題輸入.")
                        }

                        setFinalQuery(user_query);
                    } catch (e) {
                        alert(e.message);
                    }
                }}
            </button>

        </Container>
        </div>
    )

};

export default App;