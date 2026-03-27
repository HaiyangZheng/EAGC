window.HELP_IMPROVE_VIDEOJS = false;

function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    if (!dropdown || !button) return;

    const isOpen = dropdown.classList.contains('show');
    dropdown.classList.toggle('show', !isOpen);
    button.classList.toggle('active', !isOpen);
    button.setAttribute('aria-expanded', String(!isOpen));
}

function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    if (!bibtexElement || !button) return;

    const copyText = button.querySelector('.copy-text');
    const text = bibtexElement.textContent;

    navigator.clipboard.writeText(text).then(function() {
        button.classList.add('copied');
        copyText.textContent = 'Copied';

        setTimeout(function() {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    }).catch(function() {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);

        button.classList.add('copied');
        copyText.textContent = 'Copied';
        setTimeout(function() {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    });
}

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (!scrollButton) return;

    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    if (!container || !dropdown || !button) return;

    if (!container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
        button.setAttribute('aria-expanded', 'false');
    }
});

document.addEventListener('keydown', function(event) {
    if (event.key !== 'Escape') return;

    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    if (!dropdown || !button) return;

    dropdown.classList.remove('show');
    button.classList.remove('active');
    button.setAttribute('aria-expanded', 'false');
});
