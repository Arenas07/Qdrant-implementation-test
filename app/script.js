
document.addEventListener('DOMContentLoaded', function () {
    let elements = {
      modelSelect: document.getElementById('modelSelect'),
      clearChatBtn: document.getElementById('clearChatBtn'),
      messages: document.getElementById('messages'),
      composer: document.getElementById('composer'),
      prompt: document.getElementById('prompt'),
      sendBtn: document.getElementById('sendBtn')
    };
  
    let API_KEY = 'AIzaSyC_oGBVzaxQUnjO57GtTDu0gq-oU_bL4yI';
    let DEFAULT_MODEL = 'gemini-2.5-flash';
  
    // Historial de la conversación
    let conversation = [];
  
    elements.clearChatBtn.addEventListener('click', function () {
      let confirmDelete = confirm('¿Borrar toda la conversación?');
      if (!confirmDelete) {
        return;
      }
      conversation = [];
      renderMessages();
    });
  
    elements.prompt.addEventListener('input', function () {
      elements.prompt.style.height = 'auto';
      let newHeight = elements.prompt.scrollHeight;
      if (newHeight > 200) {
        newHeight = 200;
      }
      elements.prompt.style.height = newHeight + 'px';
    });
  
    elements.composer.addEventListener('submit', async function (e) {
      e.preventDefault();

      let text = elements.prompt.value?.trim();
      if (!text) return;

      // Renderizar mensaje del usuario
      appendMessage('user', text, false);
      elements.prompt.value = '';
      elements.prompt.style.height = 'auto';
      setSendingState(true);

      try {
        const res = await fetch('http://localhost:3000/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }) 
        });

        const data = await res.json();

        if (data.answer) {
          appendMessage('model', data.answer, false);
        } else {
          appendMessage('model', 'No se recibió respuesta del servidor.', true);
        }
      } catch (err) {
        appendMessage('model', 'Error de conexión con el backend.', true);
      }

      setSendingState(false);
    });

  
    function setSendingState(isSending) {
      elements.sendBtn.disabled = isSending;
      elements.prompt.disabled = isSending;
    }
  
    function appendMessage(role, text, isError) {
      conversation.push({ role: role, parts: [{ text: text }] });
      renderMessages(isError ? (conversation.length - 1) : null, isError);
      scrollToBottom();
    }
  
    function renderMessages(errorIndex, isError) {
      elements.messages.innerHTML = '';
      let i = 0;
      while (i < conversation.length) {
        let msg = conversation[i];
        let wrapper = document.createElement('div');
        wrapper.className = 'message ' + (msg.role === 'user' ? 'user' : 'model');
  
        let avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = msg.role === 'user' ? 'Tú' : 'G';
  
        let bubble = document.createElement('div');
        bubble.className = 'bubble';
        if (isError && i === errorIndex) {
          bubble.className += ' error';
        }
  
        let bubbleText = '';
        if (msg && msg.parts && msg.parts[0] && msg.parts[0].text) {
          bubbleText = msg.parts[0].text;
        }
        bubble.textContent = bubbleText;
  
        wrapper.appendChild(avatar);
        wrapper.appendChild(bubble);
        elements.messages.appendChild(wrapper);
  
        i = i + 1;
      }
    }
  
    function scrollToBottom() {
      elements.messages.scrollTop = elements.messages.scrollHeight;
    }
  
    // Llamada básica con XMLHttpRequest
    function generateContent(apiKey, model, contents, callback) {
      let url = 'https://generativelanguage.googleapis.com/v1beta/models/' + encodeURIComponent(model) + ':generateContent?key=' + encodeURIComponent(apiKey);
      let payload = { contents: contents };
  
      let xhr = new XMLHttpRequest();
      xhr.open('POST', url, true);
      xhr.setRequestHeader('Content-Type', 'application/json');
  
      xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
          if (xhr.status >= 200 && xhr.status < 300) {
            let data;
            try {
              data = JSON.parse(xhr.responseText);
            } catch (e) {
              callback('Respuesta no es JSON válido', null);
              return;
            }
  
            let text = '';
            if (data && data.candidates && data.candidates[0]) {
              let candidate = data.candidates[0];
              if (candidate.content && candidate.content.parts && candidate.content.parts[0]) {
                text = candidate.content.parts[0].text;
              }
            }
            if (!text) {
              text = '(sin contenido)';
            }
            callback(null, text);
          } else {
            let errorMsg = 'HTTP ' + xhr.status;
            try {
              let errJson = JSON.parse(xhr.responseText);
              if (errJson && errJson.error && errJson.error.message) {
                errorMsg = errorMsg + ': ' + errJson.error.message;
              }
            } catch (e2) {
              // ignore
            }
            callback(errorMsg, null);
          }
        }
      };
  
      try {
        xhr.send(JSON.stringify(payload));
      } catch (sendErr) {
        callback('No se pudo enviar la solicitud', null);
      }
    }
  });