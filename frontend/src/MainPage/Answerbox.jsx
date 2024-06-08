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
                // const embedding_response = await fetch(`${end_point}embeddings`)
                // if (!embedding_response.ok){
                //     throw new Error("Embedding failed.")
                // }
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
                <div className="container-message">查詢中...</div>
            ) : (
                llm_result ? (
                    <table className="llm-table">
                        <thead>
                            <tr>
                                <th className="col_name">資料來源</th>
                                <th className="col_name">被糾正機關</th>
                                <th className="col_name">相似度</th>
                                <th className="col_name">案由摘要</th>
                            </tr>
                        </thead>
                        <tbody>
                            {Object.keys(llm_result).map(key => (
                                <tr key={key}>
                                    <td >{llm_result[key].source}</td>
                                    <td className="summary-align">{llm_result[key].target}</td>
                                    <td className="score-align">{llm_result[key].score}</td>
                                    <td className="summary-align">{llm_result[key].summary}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <div className="container-message">查無資料</div>
                )
            )}
        </Container>
    )
};

export default AIResponse;
