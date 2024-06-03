import { useEffect, useState } from "react";
import Container from "./Container";
import config from '../../public/configs.json';
import '../../assets/css/result.css';

const end_point = config.endpoint;

const AIResponse = ({ user_query }) => {
    const [llm_result, setLlmresult] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchResult = async () => {
            try {
                setLoading(true)
                const response = await fetch(`${end_point}chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query: user_query })
                });
                if (!response.ok) {
                    throw new Error("No AI response.");
                }

                const result = await response.json();
                console.log("llm result", result);
                if (result.message === 'NoData') {
                    setLlmresult(null);
                } else {
                    setLlmresult(result);
                }
            } catch (error) {
                console.error("Error", error);
                setLlmresult(null);
            } finally {
                setLoading(false);
            }
        };

        fetchResult();
    }, [user_query]);

    return (
        <Container title="查詢結果">
            {loading ? (
                <div>查詢中...</div>
            ) : (
                llm_result ? (
                    Object.keys(llm_result).map(key => (
                        <div className="llmresult" key={key}>
                            <p><strong>資料來源:</strong> {llm_result[key].source}</p>
                            <p><strong>案由摘要:</strong> </p>
                            <p>{llm_result[key].summary}</p>
                        </div>
                    ))
                ) : (
                    <div>查無資料</div>
                )
            )}
        </Container>
    )
};

export default AIResponse;
