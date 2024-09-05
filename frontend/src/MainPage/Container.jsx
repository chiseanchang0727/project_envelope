import '../../assets/css/container.css';

const Container = ({title, children}) => {
    return (
        <div class="container">
            <div class="section">
                <h2 class="section-title">{title}</h2>
                <div class="section-content">{children}</div>
            </div>
        </div>
    );
};

export default Container;