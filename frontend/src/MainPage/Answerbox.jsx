import { useEffect, useState } from "react";
import Container from "./Container";
import config from '../../public/configs.json';
import '../../assets/css/result.css';

const end_point = config.endpoint;

const AIResponse = ({user_query}) => {
    const [llm_result, setLlmresult] = useState(null);
    // const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchResult = async () => {
            try {
                const response = await fetch(`${end_point}chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query: user_query })
                });
                if(!response.ok){
                    throw new Error("No AI response.");
                }

                const result = await response.json();
                console.log("llm result", result);
                if (result.message === 'NoData'){
                    setLlmresult(null);
                } else {
                    setLlmresult(result);
                }
            } catch (error) {
                console.error("Error", error);
            }
        };

        fetchResult();

    }, [user_query]);

    return (
        <Container title="查詢結果">
            {llm_result ? (
                <div className="llmresult">
                <p>{llm_result.ai_answer}</p>
                <p>資料來源: {llm_result.source}</p>

                
            </div>
            ) : (
                <div>
                <h3>查無資料</h3>
                </div>
            )}
        </Container>
    )    

};

export default AIResponse;
