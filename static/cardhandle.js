document.addEventListener('DOMContentLoaded', function() {
    // Modal functionality
    const modal = document.getElementById('recipe-modal');
    const modalContent = document.getElementById('modal-recipe-content');
    const closeModal = document.querySelector('.close-modal');

    // Close modal when clicking X
    closeModal.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    // Close modal when clicking outside of it
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });

    // Add event delegation for recipe cards
    document.addEventListener('click', function(e) {
        if (e.target.closest('.recipe-card')) {
            const recipeCard = e.target.closest('.recipe-card');
            const recipeData = JSON.parse(recipeCard.getAttribute('data-recipe'));
            
            // Populate the modal with full recipe data
            modalContent.innerHTML = `
                <div class="modal-recipe-header">
                    <h3>${recipeData.title || 'Món ăn không tên'}</h3>
                </div>
                ${recipeData.image ? `<img src="${recipeData.image}" alt="${recipeData.title}" class="modal-recipe-image" onerror="this.src='https://via.placeholder.com/800x400?text=Không+có+hình';this.onerror='';">` : '<div class="no-image">Không có hình</div>'}
                
                <div class="recipe-info">
                    <span class="recipe-info-label">Thời gian nấu:</span>
                    <span>${recipeData.readyInMinutes || 'N/A'} phút</span>
                </div>
                
                <div class="recipe-section">
                    <h4 class="recipe-info-label">Nguyên liệu:</h4>
                    <p>${recipeData.full_ingredients || recipeData.ingredients || 'Không có thông tin nguyên liệu'}</p>
                </div>
                
                <div class="recipe-section">
                    <h4 class="recipe-info-label">Hướng dẫn chi tiết:</h4>
                    <p>${recipeData.full_instructions || recipeData.instructions || 'Không có hướng dẫn nấu ăn'}</p>
                </div>
            `;
            
            // Show the modal
            modal.style.display = 'block';
        }
    });
});